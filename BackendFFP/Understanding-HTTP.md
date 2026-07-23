# HTTP for Backend Engineers: What's Actually Happening Under the Hood

If you've built APIs for a while, you already know the shape of HTTP: a client sends a request, a server sends back a response. But most of the interesting engineering — caching, security, cross-origin access, large file transfer — lives in the details most tutorials skip past. This post is a tour through those details, aimed at backend engineers who want more than "GET fetches data, POST creates it."

## The Two Ideas HTTP Is Built On

Before anything else, two properties define how HTTP behaves, and almost everything else in this post traces back to them.

**Statelessness.** Every HTTP request is self-contained. The server doesn't remember your last request — each one must carry everything needed to process it: headers, cookies, tokens, the full URL. This is deliberate. A stateless model means any server in a cluster can handle any request, since there's no session data pinned to a specific machine. It's a core reason horizontal scaling works as cleanly as it does. The tradeoff is that anything resembling "memory" across requests — logins, shopping carts — has to be reconstructed explicitly, usually via cookies, tokens, or server-side session stores.

**Client-server separation.** The client (browser, mobile app, another service) initiates; the server responds. The client owns presentation and interaction, the server owns resources and data. Neither needs to know the internal implementation of the other — only the contract between them.

A quick note on protocols underneath: HTTP itself doesn't dictate the transport layer, but in practice it's almost always built on **TCP**, chosen for reliability and ordered delivery over UDP's speed. That decision — and its consequences for connection setup — becomes more important as we get into performance below.

## Anatomy of a Request and a Response

A request has four parts: an HTTP method, the path/URL being requested, a set of headers, and (optionally) a body. A response mirrors this: a status code, headers, and a body.

Headers deserve real attention because they carry almost all of the metadata that makes modern HTTP work — caching rules, authentication, content negotiation, and security policy are all expressed as headers, not as special protocol features. Broadly, headers fall into a few buckets:

- **Request headers** — information about the client or its request (`User-Agent`, `Authorization`, `Accept`)
- **Response headers** — information about the server or the resource served (`Cache-Control`, `Set-Cookie`)
- **General headers** — metadata about the message itself, applicable to both directions (`Date`, `Connection`)
- **Security headers** — policy enforcement (`Content-Security-Policy`, `X-Content-Type-Options`, `Strict-Transport-Security`)

One useful mental model: headers are HTTP's extensibility mechanism. New capabilities get added to the protocol by defining new headers, not by changing the core request/response structure — which is why HTTP has stayed backward-compatible for decades while gaining huge amounts of new functionality.

## Methods Aren't Just Verbs — They're Contracts

Each HTTP method implies a contract about what kind of action is being requested, and two properties in particular are worth internalizing:

- **Safe** methods don't modify server state (GET, HEAD)
- **Idempotent** methods produce the same end-state no matter how many times you repeat them (GET, PUT, DELETE) — as opposed to **non-idempotent** methods like POST, where repeating the request creates a new effect each time (e.g., a duplicate order)

This distinction matters in practice: retry logic, client libraries, and caching layers all lean on idempotency guarantees. A client can safely retry a failed PUT without double-applying the change; retrying a failed POST without deduplication logic can create duplicate resources.

The common methods:

|Method|Purpose|Idempotent?|
|---|---|---|
|GET|Retrieve a resource|Yes|
|POST|Create a new resource|No|
|PUT|Full replacement of a resource|Yes|
|PATCH|Partial update of a resource|No|
|DELETE|Remove a resource|Yes|
|OPTIONS|Ask what the server supports for this route|Yes|

## Status Codes: A Standardized Vocabulary for Outcomes

Status codes exist so clients don't have to parse response bodies to know whether something worked. They're grouped by leading digit:

- **1xx** — informational (e.g. `100 Continue`, used when a client wants to check the server will accept a request before sending a large body)
- **2xx** — success (`200 OK`, `201 Created`, `204 No Content` — note 204 explicitly has no body)
- **3xx** — redirection (`301 Moved Permanently`, `302 Found`/temporary redirect, `304 Not Modified` — central to caching, covered below)
- **4xx** — client errors (`400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `405 Method Not Allowed`, `409 Conflict`, `413 Payload Too Large`, `429 Too Many Requests`)
- **5xx** — server errors (`500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable`, `504 Gateway Timeout`)

A distinction worth calling out explicitly: **401 vs 403**. 401 means "I don't know who you are" (missing or invalid credentials); 403 means "I know who you are, and you're not allowed to do this." Conflating them is a common API design mistake.

Another one: **502 vs 503 vs 504**, which all sound similar but describe different failure points in a proxy/load-balancer setup. 502 means the upstream server returned an invalid response; 503 means the service is intentionally unavailable (overloaded, in maintenance); 504 means the upstream simply never responded within the timeout window. If you're debugging production incidents, this distinction tells you where to look first.

## CORS: Why Your Browser Blocks Cross-Origin Requests

Cross-Origin Resource Sharing is a browser-enforced security mechanism, not a server-side restriction — the server can always serve a request, but the browser decides whether to hand the response to the requesting JavaScript. Two domains are considered different origins if they differ by scheme, domain, or port.

There are two flows:

**Simple requests** — GET/HEAD/POST with plain content types (like form-encoded or plain text) and no custom headers — go straight to the server. The server includes an `Access-Control-Allow-Origin` header in its response; the browser checks whether that header matches (or wildcards) the requesting origin. If it doesn't match, the browser blocks the JavaScript from reading the response — note that the request still happened, it's the _reading_ that's blocked.

**Preflighted requests** — triggered when a request uses a non-simple method (PUT, DELETE, PATCH), includes custom headers (like a custom `Authorization` scheme), or uses a non-simple content type (like `application/json`). Before sending the actual request, the browser sends an `OPTIONS` request asking the server: does this origin, method, and header set have permission? The server responds with `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`, and `Access-Control-Allow-Headers`. Only if all three check out does the browser send the real request. Servers can also set `Access-Control-Max-Age` to let browsers cache the preflight result and skip repeating it on every request for some period.

Almost every "CORS error" beginners hit in the browser console traces back to one of these header checks failing.

## Caching: Making the Network Layer Optional

HTTP caching exists to avoid re-transferring data that hasn't changed. The mechanism is built on a handful of headers working together:

- `Cache-Control` — sets caching policy, e.g. `max-age=10` tells the client it can reuse the response for 10 seconds without asking the server again
- `ETag` — a hash representing the current state of the resource, sent by the server
- `Last-Modified` — a timestamp of when the resource last changed

On a subsequent request, the client sends back `If-None-Match` (with the ETag it has) or `If-Modified-Since` (with the last-known timestamp). If the server determines nothing has changed, it responds with **`304 Not Modified`** and an empty body — the client just reuses its cached copy. If something changed, the server responds normally with `200` and the new data plus a new ETag.

This is entirely separate from browser HTTP caching that happens automatically for static assets — it's a mechanism your own API can implement deliberately for any resource, and it's one of the cheapest ways to cut bandwidth and latency for data that doesn't change often.

## Content Negotiation

Content negotiation lets a single URL serve different representations of the same resource depending on what the client asks for, using request headers:

- `Accept` — the format the client wants (`application/json`, `application/xml`, `text/html`)
- `Accept-Language` — preferred language
- `Accept-Encoding` — compression formats the client can decode (e.g. `gzip`)

The server picks the best match it can serve, or falls back to a default. `Accept-Encoding` in particular is worth paying attention to operationally: enabling gzip compression on responses can shrink a large JSON payload dramatically — an 11,000-row response going from several megabytes down to a fraction of that size, at the cost of a small amount of CPU on both ends to compress/decompress.

## Persistent Connections

Early HTTP (1.0) opened a new TCP connection for every single request/response — expensive, since TCP setup itself takes round trips. HTTP/1.1 made connections persistent by default: a single TCP connection can carry multiple requests and responses before being closed, controlled via the `Connection` header (`keep-alive` to persist, `close` to terminate). This single change was one of the more meaningful performance improvements in HTTP's history, because it amortizes connection-setup cost across many requests instead of paying it every time.

## Handling Large Payloads: Multipart and Streaming

Two mechanisms exist for the cases where a plain JSON body doesn't fit:

**Multipart requests** — used for file uploads. The body is split into distinct parts separated by a boundary marker (specified in the `Content-Type` header), each part carrying its own headers and a chunk of data. This is how a form can submit both structured fields and binary file data in a single request.

**Streaming responses** — used when a server wants to send data incrementally rather than assembling the entire response in memory first. The response uses `Transfer-Encoding: chunked` (or a content type like `text/event-stream`) and keeps the connection open, sending data in pieces as it becomes available, with the client appending each chunk as it arrives. This is the same underlying mechanism that powers things like server-sent events and progressive file downloads.

## A Note on TLS/SSL/HTTPS

These three terms get used loosely, so it's worth being precise: **SSL** was the original encryption protocol for securing client-server communication; it's deprecated and replaced by **TLS**, its modern successor. **HTTPS** is simply HTTP layered on top of a TLS-encrypted connection. TLS handles authentication (via certificates) and encryption, protecting data in transit from interception or tampering — none of which application code typically implements directly. It's handled below the HTTP layer, usually terminated at a load balancer, reverse proxy, or the web server itself.

## Closing Thought

None of these mechanisms are exotic — they're the stuff every backend framework implements for you, which is exactly why it's easy to go years without thinking about them directly. But debugging a mysterious CORS failure, explaining why a cache isn't invalidating, or reasoning about why an API is slow under load all come back to this same layer. Understanding it isn't about memorizing header names — it's about having a mental model of what problem each piece is solving, so when something breaks, you know which layer to look at first.

---

_Written as part of a series of notes while working through backend fundamentals from first principles._