# Security Policy

## Reporting a Vulnerability

Please use GitHub's **Private Vulnerability Reporting**
(Security tab → "Report a vulnerability") on this repository.
Do not open public issues for security reports.

## Scope

sci-nodes provides numeric filter/resample nodes only; it does not
execute code embedded in graph definitions. Note that the host runtimes
it plugs into (the nonoha engine and embedded runtime) may include
script nodes with their own threat model — see the security policy of
the host project for loading untrusted graphs.

## Supported versions

Security fixes target the latest minor release.
