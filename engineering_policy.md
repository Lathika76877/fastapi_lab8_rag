# AI Engineering Track — Security & Backend Policy

## Authentication and Token Lifecycle
All protected REST endpoints require an OAuth2 Bearer JSON Web Token issued by the /login endpoint. Access tokens remain valid for exactly 30 minutes from issuance and use HS256 symmetric signing. Inactive user accounts are blocked immediately at the dependency layer before reaching route handlers.

## Caching and Database Persistence
High-read profile endpoints use a Redis Cache-Aside strategy with a 300-second Time-To-Live (TTL) under the key namespace user:profile:{user_id}. Any PATCH or DELETE mutation evicts the cached entry immediately after committing to PostgreSQL so stale profile data is never served.

## Vector Indexing Standards
Knowledge base documents uploaded to the platform are split into 500-character overlapping segments with an 80-character overlap using RecursiveCharacterTextSplitter. Each chunk is embedded into a 384-dimensional normalized vector using all-MiniLM-L6-v2 and indexed in PostgreSQL via pgvector cosine distance.