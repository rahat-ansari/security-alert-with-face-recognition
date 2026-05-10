param(
    [string]$SourceBranch = "development-with-updated-gitignored",
    [string]$CleanBranch = "development-with-updated-gitignored-clean",
    [string]$Remote = "origin",
    [string]$BaseBranch = "main",
    [string]$CommitMessage = "Apply cleaned branch state without heavy binary history",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedGit {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args,
        [Parameter(Mandatory = $true)]
        [string]$Step
    )

    Write-Host ""
    Write-Host "[$Step]" -ForegroundColor Cyan
    Write-Host "git --no-pager $($Args -join ' ')" -ForegroundColor DarkGray

    & git --no-pager @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed at step: $Step"
    }
}

Write-Host "Starting clean-branch publish fix..." -ForegroundColor Yellow

Invoke-CheckedGit -Args @("rev-parse", "--is-inside-work-tree") -Step "Validate repository"

$workingTreeStatus = & git --no-pager status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read working tree status."
}
if ($workingTreeStatus) {
    throw "Working tree is not clean. Commit/stash changes before running this script."
}

& git --no-pager show-ref --verify --quiet "refs/heads/$SourceBranch"
if ($LASTEXITCODE -ne 0) {
    throw "Source branch '$SourceBranch' was not found locally."
}

Invoke-CheckedGit -Args @("fetch", $Remote, $BaseBranch) -Step "Fetch latest base branch"
Invoke-CheckedGit -Args @("checkout", "-B", $CleanBranch, "$Remote/$BaseBranch") -Step "Create/reset clean branch from remote base"
Invoke-CheckedGit -Args @("merge", "--squash", $SourceBranch) -Step "Squash-merge source branch changes"

& git --no-pager diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    throw "No staged changes found after squash merge; nothing to commit."
}

Invoke-CheckedGit -Args @("commit", "-m", $CommitMessage, "-m", "Co-Authored-By: Oz <oz-agent@warp.dev>") -Step "Create single clean commit"

if ($DryRun) {
    Invoke-CheckedGit -Args @("push", "--dry-run", "-u", $Remote, $CleanBranch) -Step "Dry-run push"
    Write-Host ""
    Write-Host "Dry run completed successfully. Re-run without -DryRun to publish." -ForegroundColor Green
}
else {
    Invoke-CheckedGit -Args @("push", "-u", $Remote, $CleanBranch) -Step "Push clean branch"
    Write-Host ""
    Write-Host "Done. '$CleanBranch' is published and tracking '$Remote/$CleanBranch'." -ForegroundColor Green
}
