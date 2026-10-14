# Security policy

DIANA is a research simulation package; it does not run services or handle
credentials. The realistic risks are unsafe file loading (for example a crafted
`.npz` or TIFF passed to the volume importer) and compromised dependencies.

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please **do not** open a public issue. Use GitHub's private reporting instead:
**Security → Report a vulnerability** on the repository page
(<https://github.com/OriolSansPlanell/DIANA/security/advisories/new>).
You should get an answer within two weeks. Ordinary bugs, including wrong
physics, belong in the issue tracker.
