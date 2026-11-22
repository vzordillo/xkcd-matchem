# Security Policy

## Supported Versions

Security fixes and dependency updates are provided for the current release line:

| Version | Supported |
| :--- | :--- |
| `0.1.x` | Yes |
| `< 0.1.0` | No |

---

## Reporting a Vulnerability

If you discover a security vulnerability in `xkcd-matchem` (for example, HTML
escaping or script injection in the Interactive 3D Studio export, unsafe file
parsing on untrusted chemistry files, path traversal, or workflow security
issues), **please do not report it in a public GitHub issue**.

Instead, please report it privately using **GitHub Private Vulnerability
Reporting**:

1. Open a private advisory at
   <https://github.com/vzordillo/xkcd-matchem/security/advisories/new>.
2. Include a clear description of the issue, affected version(s), minimal
   reproduction steps or proof-of-concept file, and any suggested mitigation.

### What to Expect

- **Acknowledgement:** Maintainers aim to acknowledge receipt within **7 days**.
- **Assessment & Fix:** We will reproduce the report, evaluate impact, prepare
  a patch in a private advisory branch, and coordinate a disclosure timeline
  with you before publishing a fix or advisory.
- **Credit:** Reporters who wish to be credited will be acknowledged in the
  published security advisory and `CHANGELOG.md`.

---

## Security Architecture & Built-In Safeguards

`xkcd-matchem` is designed for safe use in academic laptops, HPC login nodes,
classrooms, and static documentation sites:

1. **Offline, Local-Only Interactive 3D Studio (`viewer.html`):**
   - HTML files produced by `xm.view_structure()` and
     `xm.export_interactive_html()` are 100% self-contained and require **zero
     external network requests, CDNs, telemetry scripts, or backend servers**.
   - Local file uploads via the **Open Structure File** / drag-and-drop control
     are parsed strictly client-side in the browser via the local `FileReader`
     API — your molecular and crystal files never leave your machine.
   - Serialized JSON payloads embedded in standalone HTML escape closing
     `</script>` tags and unsafe HTML delimiters, and interactive UI controls
     populate text nodes safely rather than executing untrusted markup.
2. **Safe Text-Based File Parsing:**
   - All 29 zero-dependency native parsers in `src/xkcd_matchem/io/native/`
     parse plain text and numeric chemistry/materials formats without calling
     `eval()`, `exec()`, `pickle`, or spawning subprocesses.
   - Optional third-party adapters (`ase`, `pymatgen`, `rdkit`, `cclib`,
     `MDAnalysis`, `mdtraj`) are imported lazily only when explicitly called by
     the user.
3. **Hardened CI/CD & Release Supply Chain:**
   - All GitHub Actions workflows (`.github/workflows/ci.yml`, `docs.yml`, and
     `release.yml`) pin third-party actions to immutable 40-character full
     commit SHAs.
   - Workflow-level permissions follow least privilege (`contents: read`) with
     `persist-credentials: false` on checkout.
   - PyPI package releases use **OIDC Trusted Publishing** (`id-token: write`
     scoped exclusively to the `publish` job on tagged releases), eliminating
     long-lived PyPI API secrets.
