<#
.SYNOPSIS
  One-shot, idempotent runner for the Elias Rickensworth persona-engine
  simulation, wired to a live LM Studio server, on Windows.

.DESCRIPTION
  Fixes, in order, every snag hit so far:
    1. cargo not on PATH in the current session (stale PATH after rustup install).
    2. openssl-sys failing to build because Git's bundled MSYS Perl produces
       Unix-style paths. This does NOT just add Strawberry Perl to PATH (Git's
       perl is earlier on PATH and would still win); it sets $env:OPENSSL_SRC_PERL
       to Strawberry's perl.exe explicitly, which is the actual override the
       openssl-src build script checks -- confirmed against this repo's own
       .github/workflows/ci.yml, which does the exact same thing on their
       Windows CI runners.
    3. Wrong git branch (the persona-engine code only exists on
       claude/fauxx-persona-engine-mvp-2cchmv, not main).
    4. A placeholder / wrong / unreachable LLM model id silently degrading to
       the deterministic-only fallback without you noticing.

  Safe to re-run: every step checks state before acting.

.PARAMETER Days
  How many days to simulate. 7 = a week, 30 = a month. Default 7.

.PARAMETER Endpoint
  LM Studio's host:port. Default 169.254.83.107:1234 (your LAN server).

.PARAMETER Model
  The exact model id from LM Studio's /v1/models. Default phi-4-mini-3.8b-instruct.

.PARAMETER NoLlm
  Skip the LLM sidecar entirely (deterministic-only run), e.g. for comparison.

.PARAMETER RepoPath
  Path to the fauxx-desktop-goapbrains checkout. Default: this script's parent
  directory's sibling assumption -- override if you run this from elsewhere.

.EXAMPLE
  .\run-elias-sim.ps1 -Days 7
.EXAMPLE
  .\run-elias-sim.ps1 -Days 30 -Model qwen-3-8b-instruct
.EXAMPLE
  .\run-elias-sim.ps1 -Days 7 -NoLlm
#>

[CmdletBinding()]
param(
    [int]$Days = 7,
    [string]$Endpoint = "http://localhost:1234",          # [string]$Endpoint = "169.254.83.107:1234",
    [string]$Model = "phi-4-mini-3.8b-instruct",
    [switch]$NoLlm,
    [string]$RepoPath = "G:\Git\fauxx-desktop-goapbrains",
    [string]$ExpectedBranch = "claude/fauxx-persona-engine-mvp-2cchmv"
)

$ErrorActionPreference = "Stop"

# Normalize -Endpoint to a bare host:port regardless of what was passed in
# (with a scheme, without one, trailing slash, etc). The LM Studio check
# below prepends "http://" itself, and the Rust side does a raw TCP connect
# expecting host:port -- a leftover scheme in either place produces a
# double-scheme string like "http://http://..." that fails DNS resolution
# on the literal hostname "http".
$Endpoint = $Endpoint -replace '^https?://', '' -replace '/+$', ''

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    WARN: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "    FAIL: $msg" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# 1. cargo on PATH for this session
# ---------------------------------------------------------------------------
Write-Step "Checking cargo"
$cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargoCmd) {
    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    $cargoExe = Join-Path $cargoBin "cargo.exe"
    if (Test-Path $cargoExe) {
        $env:Path += ";$cargoBin"
        Write-Ok "cargo.exe found at $cargoExe, added its folder to PATH for this session"
    } else {
        Write-Fail "cargo.exe not found at $cargoExe. Install Rust first: https://rustup.rs (rustup-init.exe x64), then re-run this script."
        exit 1
    }
} else {
    Write-Ok "cargo already on PATH ($($cargoCmd.Source))"
}
$cargoVersion = (& cargo --version)
Write-Ok "cargo --version -> $cargoVersion"

# ---------------------------------------------------------------------------
# 2. Strawberry Perl, found (or installed) and pinned via OPENSSL_SRC_PERL
#    NOTE: this must be an OPENSSL_SRC_PERL env var, not a PATH entry --
#    Git's bundled MSYS perl is earlier on PATH and would still win a bare
#    `perl` lookup. The openssl-src build script explicitly honors
#    OPENSSL_SRC_PERL as an override, which is what actually fixes this.
# ---------------------------------------------------------------------------
Write-Step "Locating a Windows-native Perl for openssl-src"
$strawberryCandidates = @(
    "C:\Strawberry\perl\bin\perl.exe",
    (Join-Path $env:LOCALAPPDATA "Programs\Strawberry\perl\bin\perl.exe"),
    (Join-Path $env:ProgramFiles "Strawberry\perl\bin\perl.exe")
)
$perlPath = $strawberryCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $perlPath) {
    Write-Warn2 "Strawberry Perl not found in common locations. Attempting winget install (per-user, no admin needed)..."
    try {
        winget install --id StrawberryPerl.StrawberryPerl -e --accept-source-agreements --accept-package-agreements
    } catch {
        Write-Warn2 "winget invocation raised an error (see below); checking disk anyway in case it partially completed."
        Write-Host $_.Exception.Message
    }
    Start-Sleep -Seconds 2
    $perlPath = $strawberryCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $perlPath) {
    # Last resort: search the whole C:\ for it (slow, only runs if the above failed).
    Write-Warn2 "Still not found in common locations; searching C:\ (this can take a minute)..."
    $found = Get-ChildItem -Path "C:\" -Filter "perl.exe" -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "Strawberry" } | Select-Object -First 1
    if ($found) { $perlPath = $found.FullName }
}

if (-not $perlPath) {
    Write-Fail "Could not find or install Strawberry Perl. Install it manually from https://strawberryperl.com, then re-run this script."
    exit 1
}

$env:OPENSSL_SRC_PERL = $perlPath
Write-Ok "OPENSSL_SRC_PERL = $perlPath (set for this session)"

# Persist it for future sessions too, so you don't have to remember this again.
[Environment]::SetEnvironmentVariable("OPENSSL_SRC_PERL", $perlPath, "User")
Write-Ok "Also persisted OPENSSL_SRC_PERL to your user environment (future new terminals will have it automatically)"

# ---------------------------------------------------------------------------
# 3. Correct repo + branch
# ---------------------------------------------------------------------------
Write-Step "Checking repo and branch"
if (-not (Test-Path $RepoPath)) {
    Write-Fail "RepoPath '$RepoPath' does not exist. Pass -RepoPath <path> to this script."
    exit 1
}
Push-Location $RepoPath
try {
    $currentBranch = (git rev-parse --abbrev-ref HEAD 2>$null).Trim()
    if ($currentBranch -ne $ExpectedBranch) {
        Write-Warn2 "On branch '$currentBranch', expected '$ExpectedBranch'. Attempting checkout..."
        git fetch origin $ExpectedBranch
        git checkout $ExpectedBranch
        $currentBranch = (git rev-parse --abbrev-ref HEAD 2>$null).Trim()
    }
    if ($currentBranch -eq $ExpectedBranch) {
        Write-Ok "On branch $currentBranch"
    } else {
        Write-Fail "Still not on $ExpectedBranch (got '$currentBranch'). Fix manually: git checkout $ExpectedBranch"
        exit 1
    }

    $examplePath = "crates\fauxx-core\examples\day_in_the_life.rs"
    $exampleContent = Get-Content $examplePath -Raw -ErrorAction SilentlyContinue
    if ($exampleContent -notmatch "FAUXX_LLM") {
        Write-Fail "$examplePath doesn't contain the LLM patch (no FAUXX_LLM reference). Copy day_in_the_life_llm.rs over it first."
        exit 1
    }
    Write-Ok "$examplePath has the LLM patch"
} finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
# 4. LM Studio reachability + model id sanity check (skipped if -NoLlm)
# ---------------------------------------------------------------------------
if (-not $NoLlm) {
    Write-Step "Checking LM Studio at $Endpoint"
    try {
        $models = Invoke-RestMethod -Uri "http://$Endpoint/v1/models" -TimeoutSec 5
        $ids = $models.data | ForEach-Object { $_.id }
        if ($ids -contains $Model) {
            Write-Ok "Model '$Model' is loaded and reachable at $Endpoint"
        } else {
            Write-Fail "Model '$Model' is NOT in LM Studio's loaded model list. Available: $($ids -join ', ')"
            Write-Fail "Fix: pass -Model <one of the above>, or load '$Model' in LM Studio first."
            exit 1
        }
    } catch {
        Write-Fail "Could not reach LM Studio at http://$Endpoint/v1/models -- $($_.Exception.Message)"
        Write-Fail "Check LM Studio's server is running and 'Serve on Local Network' is on, or re-run with -NoLlm for a deterministic-only run."
        exit 1
    }
} else {
    Write-Step "Skipping LM Studio check (-NoLlm): deterministic-only run"
}

# ---------------------------------------------------------------------------
# 5. Set the FAUXX_* env vars and run
# ---------------------------------------------------------------------------
Write-Step "Configuring simulation ($Days day(s)$(if ($NoLlm) { ', LLM disabled' } else { ", LLM via $Model @ $Endpoint" }))"
if ($NoLlm) {
    $env:FAUXX_LLM = "0"
} else {
    $env:FAUXX_LLM = "1"
    $env:FAUXX_LLM_ENDPOINT = $Endpoint
    $env:FAUXX_LLM_MODEL = $Model
}
$env:FAUXX_SIM_DAYS = "$Days"

Push-Location $RepoPath
try {
    $logDir = Join-Path $RepoPath "sim-logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $logFile = Join-Path $logDir "elias-$Days`d-$stamp.log"

    Write-Step "Running: cargo run -p fauxx-core --example day_in_the_life"
    Write-Host "    (first run compiles the whole crate + vendored SQLCipher/OpenSSL -- can take several minutes)`n"

    # IMPORTANT: cargo (like most Rust/Go tooling) writes its normal build
    # progress ("Compiling ...", "Finished ...") to STDERR by convention, not
    # because anything is wrong. With 2>&1 merging streams, PowerShell wraps
    # each stderr line in an ErrorRecord, and the top-level
    # $ErrorActionPreference = "Stop" would treat the very first one as a
    # terminating error -- killing the run before cargo even gets a chance to
    # succeed or fail for real. Scope it to "Continue" just for this call, and
    # check $LASTEXITCODE explicitly afterward instead.
    #
    #   $prevEAP = $ErrorActionPreference
    #   $ErrorActionPreference = "Continue"
    #   cargo run -p fauxx-core --example day_in_the_life 2>&1 | Tee-Object -FilePath $logFile
    #   $exitCode = $LASTEXITCODE
    #   $ErrorActionPreference = $prevEAP
    #   $prevEAP = $ErrorActionPreference

    $ErrorActionPreference = "Continue"
    & cargo run -p fauxx-core --example day_in_the_life *> $logFile
    $exitCode = $LASTEXITCODE
    Get-Content $logFile
    $ErrorActionPreference = $prevEAP

    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Ok "Transcript saved to $logFile"
    } else {
        Write-Fail "cargo exited with code $exitCode. Full transcript (incl. the real error) is in $logFile"
    }
} finally {
    Pop-Location
}