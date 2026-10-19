// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Decoding of the responses to requests that modify a service.
//!
//! These functions take the parts of an HTTP response, so any
//! [`HttpClient`](crate::HttpClient) decodes with the same rules, vendor
//! fixes included. The reqwest client uses them for its own responses.

use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use http::header;
use http::HeaderMap;
use http::StatusCode;
use nv_redfish_core::AsyncTask;
use nv_redfish_core::BmcError;
use nv_redfish_core::BmcErrorClass;
use nv_redfish_core::DecodeError;
use nv_redfish_core::ModificationResponse;
use nv_redfish_core::ODataETag;
use nv_redfish_core::ODataId;
use serde::de::DeserializeOwned;
use serde_json::Value;
use url::Url;

use crate::schema::redfish::message::Message;
use crate::schema::redfish::redfish_error::RedfishErrorContents;
use crate::schema::redfish::resource::Health;

/// A response that does not decode as the outcome of a request.
#[derive(Debug)]
pub enum ModificationDecodeError {
    /// The status, a header or the body is not one the request allows.
    InvalidResponse {
        /// URL of the response.
        url: Url,
        /// Status of the response.
        status: StatusCode,
        /// What is wrong, or the body of an unsuccessful response.
        text: String,
    },
    /// The body is not JSON.
    Json(serde_json::Error),
    /// The body deserialized neither into the requested type nor as a
    /// success message.
    Decode(DecodeError),
}

impl fmt::Display for ModificationDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResponse { url, status, text } => write!(
                f,
                "Invalid HTTP response - url: {url} status: {status} text: {text}"
            ),
            Self::Json(e) => write!(f, "JSON Decode error: {e}"),
            Self::Decode(e) => write!(
                f,
                "JSON deserialization error at line {} column {} path {}: {e}",
                e.inner().line(),
                e.inner().column(),
                e.path(),
            ),
        }
    }
}

impl StdError for ModificationDecodeError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::InvalidResponse { .. } => None,
            Self::Json(e) => Some(e),
            Self::Decode(e) => Some(e.inner()),
        }
    }
}

impl BmcError for ModificationDecodeError {
    fn error_class(&self) -> BmcErrorClass {
        match self {
            Self::InvalidResponse { status, .. } => BmcErrorClass::HttpResponse {
                status: status.as_u16(),
            },
            Self::Json(_) | Self::Decode(_) => BmcErrorClass::ResponseParse,
        }
    }

    fn into_decode_error(self) -> Result<DecodeError, Self> {
        match self {
            Self::Decode(error) => Ok(error),
            other => Err(other),
        }
    }
}

/// The outcome that a Redfish success message reports, by the key of its
/// message in the Base message registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuccessCode {
    /// `Success`: the request completed.
    Success,
    /// `Created`: the request created a resource.
    Created,
    /// `NoOperation`: the request changed nothing, because nothing needed
    /// to change.
    NoOperation,
}

impl SuccessCode {
    /// The code a message ID or an error `code` names by its last segment,
    /// as in `Base.1.8.Success`, whatever its registry and version.
    fn of(message_id: &str) -> Option<Self> {
        match message_id
            .rsplit_once('.')
            .map_or(message_id, |(_, key)| key)
        {
            "Success" => Some(Self::Success),
            "Created" => Some(Self::Created),
            "NoOperation" => Some(Self::NoOperation),
            _ => None,
        }
    }
}

/// The success that a response body reports when it is a success message
/// instead of a resource or an action's result.
///
/// DSP0266 lets an action without a result answer with an error-shaped
/// body whose message indicates success, and services answer other requests
/// with messages alone too:
///
/// - A body with `error`, or with `Success` as Supermicro BMCs use, reports
///   that object's `code`, as in `{"error": {"code": "Base.1.8.Success",
///   "message": "..."}}`. Any other code is not a success, even when the
///   body also has `@Message.ExtendedInfo`.
/// - A body with only `@Message.ExtendedInfo` is [`SuccessCode::Created`]
///   when one of its messages is `Created`, [`SuccessCode::NoOperation`]
///   when all of them are `NoOperation`, and otherwise
///   [`SuccessCode::Success`].
///
/// A message with a `Critical` `MessageSeverity` or `Severity` makes the body
/// not a success.
#[must_use]
pub fn success_code(body: &Value) -> Option<SuccessCode> {
    /// The messages of a body or of its status object.
    #[derive(serde::Deserialize)]
    struct Messages {
        #[serde(rename = "@Message.ExtendedInfo", default)]
        extended_info: Option<Vec<Message>>,
    }

    let messages = |value: &Value| {
        <Messages as serde::Deserialize>::deserialize(value)
            .ok()?
            .extended_info
    };
    let is_critical = |messages: &[Message]| {
        messages.iter().any(|message| {
            message.message_severity == Some(Health::Critical)
                || message.severity.as_deref() == Some("Critical")
        })
    };

    let body_messages = messages(body);
    if body_messages.as_deref().is_some_and(is_critical) {
        return None;
    }

    if let Some(status) = body.get("error").or_else(|| body.get("Success")) {
        let contents = <RedfishErrorContents as serde::Deserialize>::deserialize(status).ok()?;
        if messages(status).as_deref().is_some_and(is_critical) {
            return None;
        }
        return SuccessCode::of(&contents.code);
    }

    let body_messages = body_messages?;
    let codes = || {
        body_messages
            .iter()
            .map(|message| SuccessCode::of(&message.message_id))
    };
    if codes().any(|code| code == Some(SuccessCode::Created)) {
        Some(SuccessCode::Created)
    } else if !body_messages.is_empty()
        && codes().all(|code| code == Some(SuccessCode::NoOperation))
    {
        Some(SuccessCode::NoOperation)
    } else {
        Some(SuccessCode::Success)
    }
}

/// Decodes the response to a request that modifies a service: an action, a
/// create, an update, a delete or an upload.
///
/// `url` is the URL of the response, after any redirect, which a relative
/// `Location` resolves against. `body` is the whole response body.
///
/// - `204 No Content` is [`ModificationResponse::Empty`].
/// - `202 Accepted` is a [`ModificationResponse::Task`] at its `Location`,
///   to poll after its `Retry-After` seconds.
/// - `200 OK` or `201 Created` is the body as `T`, with the `ETag` header as
///   the `@odata.etag` of a body that names its `@odata.id`. A body that is
///   not `T` but a success message (see [`success_code`]) is
///   [`ModificationResponse::Empty`], or the resource its `Location` names
///   when that is a `T`. An empty body is that resource, or
///   [`ModificationResponse::Empty`] without a `Location`.
///
/// # Errors
///
/// - [`ModificationDecodeError::InvalidResponse`] for an unsuccessful status,
///   with the body as its text; another successful status; a `202` without
///   `Location`; or a `Location` that does not name a resource of the same
///   origin.
/// - [`ModificationDecodeError::Json`] when a `200` or `201` body is not JSON.
/// - [`ModificationDecodeError::Decode`] when a `200` or `201` body is neither
///   `T` nor a success message, or an empty one names a `Location` that is not
///   a `T`.
pub fn decode_modification<T>(
    url: &Url,
    status: StatusCode,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<ModificationResponse<T>, ModificationDecodeError>
where
    T: DeserializeOwned,
{
    if !status.is_success() {
        return Err(ModificationDecodeError::InvalidResponse {
            url: url.clone(),
            status,
            text: String::from_utf8_lossy(body).into_owned(),
        });
    }

    // Resolve the header once, but defer propagating its error until a
    // status branch actually uses Location. A malformed, irrelevant
    // Location must not turn a valid 204 or entity-bearing 200/201 into an
    // error.
    let location = location_from_headers(headers, url, status);

    match status {
        StatusCode::NO_CONTENT => Ok(ModificationResponse::Empty),
        StatusCode::ACCEPTED => {
            let Some(task_location) = location? else {
                return Err(ModificationDecodeError::InvalidResponse {
                    url: url.clone(),
                    status,
                    text: String::from("202 Accepted without Location header"),
                });
            };

            Ok(ModificationResponse::Task(AsyncTask {
                location: task_location.into(),
                retry_after: retry_after_from_headers(headers),
            }))
        }
        StatusCode::OK | StatusCode::CREATED => {
            if !body.is_empty() {
                let mut value: Value =
                    serde_json::from_slice(body).map_err(ModificationDecodeError::Json)?;

                if value.get("@odata.id").is_some() {
                    if let Some(etag) = etag_from_headers(headers) {
                        inject_etag(&etag, &mut value);
                    }
                }

                // Non-empty 200/201 bodies are typed responses selected by the caller.
                //
                // - These bodies are not required to be Redfish resources with @odata.id.
                //
                // - DSP0266 POST actions with no response body may still return "an error
                //   response, with a message that indicates success".
                return match serde_path_to_error::deserialize(&value) {
                    // Non-empty 200/201 body matched the caller-selected type.
                    Ok(entity) => Ok(ModificationResponse::Entity(entity)),
                    Err(err) => {
                        if success_code(&value).is_some() {
                            // A success message is not the requested
                            // representation. Preserve an accompanying
                            // resource Location in the same way as an
                            // empty 200/201 response.
                            if let Some(location) = location? {
                                return Ok(entity_reference(&location)
                                    .map_or(ModificationResponse::Empty, |entity| {
                                        ModificationResponse::Entity(entity)
                                    }));
                            }
                            Ok(ModificationResponse::Empty)
                        } else {
                            // The response was successful JSON, but it did
                            // not match the caller-selected response type.
                            Err(ModificationDecodeError::Decode(err))
                        }
                    }
                };
            }

            if let Some(location) = location? {
                return entity_reference(&location)
                    .map(ModificationResponse::Entity)
                    .map_err(ModificationDecodeError::Decode);
            }

            Ok(ModificationResponse::Empty)
        }
        _ => Err(ModificationDecodeError::InvalidResponse {
            url: url.clone(),
            status,
            text: format!("Unexpected successful status code: {status}"),
        }),
    }
}

/// Decodes the response to polling the URI of an asynchronous operation.
///
/// A `202 Accepted` is still running: a [`ModificationResponse::Task`] at its
/// `Location`, or at the polled URI when it names none, as pending responses
/// often omit it. A `204 No Content` naming a `Location` is that resource, as
/// an empty `200` with one is. Any other response decodes as
/// [`decode_modification`] does.
///
/// # Errors
///
/// As [`decode_modification`], and
/// [`ModificationDecodeError::InvalidResponse`] when the `Location` of a
/// `202` does not name a resource of the same origin.
pub fn decode_poll<T>(
    url: &Url,
    status: StatusCode,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<ModificationResponse<T>, ModificationDecodeError>
where
    T: DeserializeOwned,
{
    match status {
        StatusCode::ACCEPTED => {
            let location = location_from_headers(headers, url, status)?
                .unwrap_or_else(|| odata_path_from_url(url).into());
            Ok(ModificationResponse::Task(AsyncTask {
                location: location.into(),
                retry_after: retry_after_from_headers(headers),
            }))
        }
        StatusCode::NO_CONTENT => location_from_headers(headers, url, status)?.map_or(
            Ok(ModificationResponse::Empty),
            |location| {
                entity_reference(&location)
                    .map(ModificationResponse::Entity)
                    .map_err(ModificationDecodeError::Decode)
            },
        ),
        _ => decode_modification(url, status, headers, body),
    }
}

/// A `T` that is only a reference to the resource at `location`.
fn entity_reference<T: DeserializeOwned>(location: &ODataId) -> Result<T, DecodeError> {
    serde_path_to_error::deserialize(serde_json::json!({ "@odata.id": location }))
}

/// Resolve a Redfish `Location` header into a same-origin path and query.
///
/// HTTP defines `Location` as a URI reference, so values may be absolute,
/// root-relative, path-relative, or query-only. Resolution must use the final
/// response URL; treating a relative value as a path rooted at the configured
/// BMC endpoint can target a different resource. The returned `ODataId` keeps
/// only path and query because fragments are not sent in subsequent HTTP
/// requests and transport always uses the configured BMC origin.
pub fn location_from_headers(
    headers: &HeaderMap,
    response_url: &Url,
    status: StatusCode,
) -> Result<Option<ODataId>, ModificationDecodeError> {
    let invalid_response = |text: &'static str| ModificationDecodeError::InvalidResponse {
        url: response_url.clone(),
        status,
        text: text.to_string(),
    };

    let Some(value) = headers.get(header::LOCATION) else {
        return Ok(None);
    };

    let raw = value
        .to_str()
        .map_err(|_| invalid_response("Location header is not valid text"))?;

    let raw = raw.trim();

    // Joining either value would resolve back to the response resource, which
    // cannot identify a newly created session or asynchronous task monitor.
    if raw.is_empty() || raw.starts_with('#') {
        return Err(invalid_response(
            "Location header does not identify a resource",
        ));
    }

    let resolved = response_url
        .join(raw)
        .map_err(|_| invalid_response("Location header is not a valid URI reference"))?;

    // Redfish follow-up requests carry BMC credentials. Reject another origin
    // before reducing the URL to an OData path and losing that distinction.
    if resolved.origin() != response_url.origin() {
        return Err(invalid_response(
            "Location header resolves to a different origin",
        ));
    }

    Ok(Some(odata_path_from_url(&resolved).into()))
}

fn odata_path_from_url(url: &Url) -> String {
    let mut path_and_query = url.path().to_string();

    // Preserve the query separately from the path so later polling or deletion
    // sends it as a query instead of percent-encoded path text.
    if let Some(query) = url.query() {
        path_and_query.push('?');
        path_and_query.push_str(query);
    }

    path_and_query
}

pub fn etag_from_headers(headers: &HeaderMap) -> Option<ODataETag> {
    headers
        .get(header::ETAG)
        .and_then(|value| value.to_str().ok())
        .map(|v| v.to_string().into())
}

fn retry_after_from_headers(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        // RFC 9110 defines the numeric Retry-After form as delay-seconds.
        // This helper handles that form and leaves HTTP-date support out of scope.
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

pub fn inject_etag(etag: &ODataETag, body: &mut Value) {
    if let Some(obj) = body.as_object_mut() {
        let etag_value = Value::String(etag.to_string());

        // Handles both absent and null values
        obj.entry("@odata.etag")
            .and_modify(|v| *v = etag_value.clone())
            .or_insert(etag_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::error::Error as StdError;

    use http::HeaderValue;
    use serde_json::json;

    /// A resource reference, as a create that names its `Location` returns.
    #[derive(Debug, serde::Deserialize)]
    struct Reference {
        #[serde(rename = "@odata.id")]
        odata_id: String,
    }

    /// A resource with an ETag.
    #[derive(Debug, serde::Deserialize)]
    struct Resource {
        #[serde(rename = "@odata.id")]
        odata_id: String,
        #[serde(rename = "@odata.etag")]
        odata_etag: Option<String>,
    }

    fn url() -> Url {
        Url::parse("https://bmc.example/redfish/v1/Systems/1/Actions/ComputerSystem.Reset")
            .expect("valid URL")
    }

    fn headers(pairs: &[(header::HeaderName, &'static str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|(name, value)| (name.clone(), HeaderValue::from_static(value)))
            .collect()
    }

    fn decode<T: DeserializeOwned>(
        status: StatusCode,
        headers: &HeaderMap,
        body: &Value,
    ) -> Result<ModificationResponse<T>, ModificationDecodeError> {
        decode_modification(&url(), status, headers, body.to_string().as_bytes())
    }

    #[test]
    fn success_messages_report_their_code() {
        for (body, code) in [
            // DSP0266: an error response whose message indicates success.
            (
                json!({"error": {"code": "Base.1.8.Success", "message": "Done."}}),
                Some(SuccessCode::Success),
            ),
            // Supermicro: the status under `Success`.
            (
                json!({"Success": {"code": "Base.1.10.Success", "message": "Done."}}),
                Some(SuccessCode::Success),
            ),
            (
                json!({"error": {"code": "Base.1.8.Created", "message": "Created."}}),
                Some(SuccessCode::Created),
            ),
            (
                json!({"error": {"code": "Base.1.8.NoOperation", "message": "Nothing."}}),
                Some(SuccessCode::NoOperation),
            ),
            (
                json!({"error": {"code": "Base.1.8.GeneralError", "message": "Failed."}}),
                None,
            ),
            // A status object without its required message is not one.
            (json!({"Success": {"code": "Base.1.10.Success"}}), None),
            (json!({"Success": "Base.1.10.Success"}), None),
            // Extended information alone.
            (
                json!({"@Message.ExtendedInfo": [{
                    "MessageId": "Base.1.13.0.Success",
                    "MessageSeverity": "OK"
                }]}),
                Some(SuccessCode::Success),
            ),
            (
                json!({"@Message.ExtendedInfo": [{"MessageId": "IDRAC.2.9.SYS413"}]}),
                Some(SuccessCode::Success),
            ),
            (
                json!({"@Message.ExtendedInfo": [
                    {"MessageId": "Base.1.8.Success"},
                    {"MessageId": "Base.1.8.Created"}
                ]}),
                Some(SuccessCode::Created),
            ),
            (
                json!({"@Message.ExtendedInfo": [{"MessageId": "Base.1.8.NoOperation"}]}),
                Some(SuccessCode::NoOperation),
            ),
            (
                json!({"@Message.ExtendedInfo": [
                    {"MessageId": "Base.1.8.NoOperation"},
                    {"MessageId": "Base.1.8.Success"}
                ]}),
                Some(SuccessCode::Success),
            ),
            (
                json!({"@Message.ExtendedInfo": []}),
                Some(SuccessCode::Success),
            ),
            // Messages need a MessageId.
            (
                json!({"@Message.ExtendedInfo": [{"Message": "Done."}]}),
                None,
            ),
            (json!({"Result": "Accepted"}), None),
            (json!(null), None),
        ] {
            assert_eq!(success_code(&body), code, "{body}");
        }
    }

    #[test]
    fn a_critical_message_or_an_error_code_is_not_a_success() {
        for body in [
            json!({"@Message.ExtendedInfo": [{
                "MessageId": "Base.1.8.InternalError",
                "MessageSeverity": "Critical"
            }]}),
            json!({"@Message.ExtendedInfo": [
                {"MessageId": "Base.1.8.Success", "Severity": "OK"},
                {"MessageId": "Base.1.8.InternalError", "Severity": "Critical"}
            ]}),
            json!({"error": {
                "code": "Base.1.8.Success",
                "message": "Done.",
                "@Message.ExtendedInfo": [{
                    "MessageId": "Base.1.8.InternalError",
                    "MessageSeverity": "Critical"
                }]
            }}),
            json!({
                "error": {"code": "Base.1.8.Success", "message": "Done."},
                "@Message.ExtendedInfo": [{
                    "MessageId": "Base.1.8.InternalError",
                    "Severity": "Critical"
                }]
            }),
            // The code of an error decides, whatever the messages beside it.
            json!({
                "error": {"code": "Base.1.8.GeneralError", "message": "Failed."},
                "@Message.ExtendedInfo": [{"MessageId": "Base.1.8.Success"}]
            }),
        ] {
            assert_eq!(success_code(&body), None, "{body}");
        }
    }

    #[test]
    fn a_success_message_is_an_empty_response() -> Result<(), Box<dyn StdError>> {
        for body in [
            json!({"error": {"code": "Base.1.8.Success", "message": "Done."}}),
            json!({"Success": {"code": "Base.1.10.Success", "message": "Done."}}),
            json!({"@Message.ExtendedInfo": [{"MessageId": "Base.1.13.0.Success"}]}),
        ] {
            let response = decode::<()>(StatusCode::OK, &HeaderMap::new(), &body)?;
            assert!(matches!(response, ModificationResponse::Empty), "{}", body);
        }
        Ok(())
    }

    #[test]
    fn a_body_that_is_neither_the_result_nor_a_success_message_fails() {
        for body in [
            json!({"Result": "Accepted"}),
            json!({"error": {"code": "Base.1.8.GeneralError", "message": "Failed."}}),
            json!({"@Message.ExtendedInfo": [{
                "MessageId": "Base.1.8.InternalError",
                "MessageSeverity": "Critical"
            }]}),
        ] {
            let error =
                decode::<()>(StatusCode::OK, &HeaderMap::new(), &body).expect_err("not a success");
            assert!(
                matches!(error, ModificationDecodeError::Decode(_)),
                "{}: {}",
                body,
                error
            );
            assert_eq!(error.error_class(), BmcErrorClass::ResponseParse);
        }

        let error = decode_modification::<()>(&url(), StatusCode::OK, &HeaderMap::new(), b"{")
            .expect_err("not JSON");
        assert!(matches!(error, ModificationDecodeError::Json(_)));
    }

    #[test]
    fn a_success_message_with_a_location_is_that_resource() -> Result<(), Box<dyn StdError>> {
        let location = headers(&[(header::LOCATION, "/redfish/v1/TaskService/Tasks/42")]);
        let body = json!({"@Message.ExtendedInfo": [{"MessageId": "Base.1.12.Success"}]});

        let response = decode::<Reference>(StatusCode::OK, &location, &body)?;
        let ModificationResponse::Entity(reference) = response else {
            return Err("expected the resource the Location names".into());
        };
        assert_eq!(reference.odata_id, "/redfish/v1/TaskService/Tasks/42");

        // A result type that is not a reference ignores the Location.
        let response = decode::<()>(StatusCode::OK, &location, &body)?;
        assert!(matches!(response, ModificationResponse::Empty));
        Ok(())
    }

    #[test]
    fn statuses_without_a_result_decode_by_their_headers() -> Result<(), Box<dyn StdError>> {
        let response =
            decode_modification::<()>(&url(), StatusCode::NO_CONTENT, &HeaderMap::new(), b"")?;
        assert!(matches!(response, ModificationResponse::Empty));

        let accepted = headers(&[
            (header::LOCATION, "/redfish/v1/TaskService/Tasks/7"),
            (header::RETRY_AFTER, "5"),
        ]);
        let ModificationResponse::Task(task) =
            decode_modification::<()>(&url(), StatusCode::ACCEPTED, &accepted, b"")?
        else {
            return Err("202 is a task".into());
        };
        assert_eq!(
            task.location.0.to_string(),
            "/redfish/v1/TaskService/Tasks/7"
        );
        assert_eq!(task.retry_after, Some(Duration::from_secs(5)));

        let error = decode_modification::<()>(&url(), StatusCode::ACCEPTED, &HeaderMap::new(), b"")
            .expect_err("202 without Location");
        assert!(matches!(
            error,
            ModificationDecodeError::InvalidResponse { ref text, .. }
                if text == "202 Accepted without Location header"
        ));

        let created = headers(&[(header::LOCATION, "Tasks/8")]);
        let ModificationResponse::Entity(reference) =
            decode_modification::<Reference>(&url(), StatusCode::CREATED, &created, b"")?
        else {
            return Err("an empty 201 with a Location is that resource".into());
        };
        assert_eq!(reference.odata_id, "/redfish/v1/Systems/1/Actions/Tasks/8");
        Ok(())
    }

    #[test]
    fn unexpected_and_unsuccessful_statuses_are_invalid_responses() {
        let error =
            decode_modification::<()>(&url(), StatusCode::PARTIAL_CONTENT, &HeaderMap::new(), b"")
                .expect_err("206");
        assert!(matches!(
            error,
            ModificationDecodeError::InvalidResponse { ref text, .. }
                if text == "Unexpected successful status code: 206 Partial Content"
        ));

        let error = decode_modification::<()>(
            &url(),
            StatusCode::BAD_REQUEST,
            &HeaderMap::new(),
            br#"{"error":{"code":"Base.1.8.ActionParameterUnknown","message":"Bad."}}"#,
        )
        .expect_err("400");
        assert!(matches!(
            error,
            ModificationDecodeError::InvalidResponse { ref text, .. }
                if text.contains("ActionParameterUnknown")
        ));
        assert_eq!(
            error.error_class(),
            BmcErrorClass::HttpResponse { status: 400 }
        );
    }

    #[test]
    fn the_etag_header_is_the_etag_of_a_returned_resource() -> Result<(), Box<dyn StdError>> {
        let etag = headers(&[(header::ETAG, "\"7\"")]);
        let ModificationResponse::Entity(resource) = decode::<Resource>(
            StatusCode::OK,
            &etag,
            &json!({"@odata.id": "/redfish/v1/AccountService"}),
        )?
        else {
            return Err("expected the resource".into());
        };
        assert_eq!(resource.odata_id, "/redfish/v1/AccountService");
        assert_eq!(resource.odata_etag.as_deref(), Some("\"7\""));
        Ok(())
    }

    #[test]
    fn a_pending_poll_without_a_location_stays_at_the_polled_uri() -> Result<(), Box<dyn StdError>>
    {
        let polled = Url::parse("https://bmc.example/redfish/v1/TaskService/TaskMonitors/1?x=y")?;
        let ModificationResponse::Task(task) =
            decode_poll::<()>(&polled, StatusCode::ACCEPTED, &HeaderMap::new(), b"")?
        else {
            return Err("202 is still running".into());
        };
        assert_eq!(
            task.location.0.to_string(),
            "/redfish/v1/TaskService/TaskMonitors/1?x=y"
        );

        let created = headers(&[(header::LOCATION, "/redfish/v1/Systems/1")]);
        let ModificationResponse::Entity(reference) =
            decode_poll::<Reference>(&polled, StatusCode::NO_CONTENT, &created, b"")?
        else {
            return Err("a 204 with a Location is that resource".into());
        };
        assert_eq!(reference.odata_id, "/redfish/v1/Systems/1");

        let response = decode_poll::<()>(
            &polled,
            StatusCode::OK,
            &HeaderMap::new(),
            br#"{"Success":{"code":"Base.1.10.Success","message":"Done."}}"#,
        )?;
        assert!(matches!(response, ModificationResponse::Empty));
        Ok(())
    }

    #[test]
    fn location_header_resolves_valid_and_rejects_invalid_references(
    ) -> Result<(), Box<dyn StdError>> {
        struct TestCase(&'static str, Result<&'static str, &'static str>);

        let response_url = Url::parse("https://bmc.example/redfish/v1/SessionService/Sessions")?;

        let cases = [
            TestCase(
                "https://bmc.example/redfish/v1/SessionService/Sessions/1?token=abc",
                Ok("/redfish/v1/SessionService/Sessions/1?token=abc"),
            ),
            TestCase(
                "/redfish/v1/SessionService/Sessions/1?token=abc",
                Ok("/redfish/v1/SessionService/Sessions/1?token=abc"),
            ),
            TestCase(
                "Sessions/1?token=abc",
                Ok("/redfish/v1/SessionService/Sessions/1?token=abc"),
            ),
            TestCase(
                "../TaskService/Tasks/42?token=abc",
                Ok("/redfish/v1/TaskService/Tasks/42?token=abc"),
            ),
            TestCase(
                "?token=abc",
                Ok("/redfish/v1/SessionService/Sessions?token=abc"),
            ),
            TestCase(
                "//bmc.example/redfish/v1/TaskService/Tasks/42",
                Ok("/redfish/v1/TaskService/Tasks/42"),
            ),
            TestCase("", Err("Location header does not identify a resource")),
            TestCase(
                "#fragment",
                Err("Location header does not identify a resource"),
            ),
            TestCase(
                "https://other.example/redfish/v1/TaskService/Tasks/42",
                Err("Location header resolves to a different origin"),
            ),
            TestCase(
                "//bmc.example:8443/redfish/v1/TaskService/Tasks/42",
                Err("Location header resolves to a different origin"),
            ),
            TestCase(
                "//host:99999/path",
                Err("Location header is not a valid URI reference"),
            ),
        ];

        for TestCase(raw, expected) in cases {
            let mut headers = HeaderMap::new();
            headers.insert(header::LOCATION, raw.parse::<HeaderValue>()?);

            let result = location_from_headers(&headers, &response_url, StatusCode::CREATED);

            match (result, expected) {
                (Ok(Some(location)), Ok(expected)) => {
                    assert_eq!(location.to_string(), expected, "{raw}");
                }
                (Err(ModificationDecodeError::InvalidResponse { text, .. }), Err(expected)) => {
                    assert_eq!(text, expected, "{raw}");
                }
                _ => return Err(format!("{raw}: unexpected Location result").into()),
            }
        }

        Ok(())
    }
}
