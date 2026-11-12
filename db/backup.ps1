param(
    [string]$BackupDir = 'G:\pg_backups',
    [string]$PgPassFile = 'G:\pg_backups\pgpass.conf',
    [int]$RetentionDays = 7
)

# Two kinds of dump:
#   research_<stamp>.dump          every night: everything except the rows of transcript_files
#   transcript_files_<stamp>.dump  only when transcript_files has gained rows; one copy is kept
# transcript_files holds the raw vendor XMLs and is append-only, so its rows do not need
# copying again until a load adds some. Restoring takes both dumps: see db/README.md.

$ErrorActionPreference = 'Stop'
$pgBin = 'C:\Program Files\PostgreSQL\18\bin'
$connection = @('-h', 'localhost', '-p', '5432', '-U', 'postgres', '-d', 'research', '-w')
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

$stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$log = Join-Path $BackupDir 'backup.log'
$stateFile = Join-Path $BackupDir 'transcript_files.state'
$partial = $null

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) $Message"
}

function Invoke-Dump([string]$Final, [string[]]$Options) {
    $target = "$Final.partial"
    $script:partial = $target  # so the catch block can remove a half-written dump
    $arguments = $connection + @('--format=custom', '--compress=1') + $Options + @("--file=$target")
    $messages = & (Join-Path $pgBin 'pg_dump.exe') @arguments 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "pg_dump exited with code $LASTEXITCODE. $($messages.Trim())"
    }
    if ((Get-Item -LiteralPath $target).Length -eq 0) {
        throw 'pg_dump produced an empty file'
    }
    Move-Item -LiteralPath $target -Destination $Final
    $script:partial = $null
}

try {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $env:PGPASSFILE = $PgPassFile
    Write-Log "START user=$identity appdata=$env:APPDATA userprofile=$env:USERPROFILE pgpass_path=$PgPassFile pgpass_exists=$(Test-Path -LiteralPath $PgPassFile)"
    if (-not (Test-Path -LiteralPath $PgPassFile)) {
        try { Get-Item -LiteralPath $PgPassFile -ErrorAction Stop | Out-Null }
        catch { Write-Log "PGPASS_ERROR $($_.Exception.Message)" }
    }

    # 1. The nightly dump: every table's rows except transcript_files (its definition is included)
    $final = Join-Path $BackupDir "research_$stamp.dump"
    Invoke-Dump $final @('--exclude-table-data=public.transcript_files')

    $cutoff = (Get-Date).AddDays(-$RetentionDays)
    Get-ChildItem -LiteralPath $BackupDir -Filter 'research_*.dump' -File |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        Remove-Item -Force
    Write-Log "OK $final"

    # 2. The transcript rows, only if the table changed since they were last dumped
    $query = "SELECT count(*) || ' rows, max id ' || coalesce(max(id), 0) FROM public.transcript_files"
    $state = (& (Join-Path $pgBin 'psql.exe') @connection -At -c $query 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "could not read transcript_files: $state"
    }
    $previous = if (Test-Path -LiteralPath $stateFile) { (Get-Content -LiteralPath $stateFile -Raw).Trim() } else { '' }
    $existing = @(Get-ChildItem -LiteralPath $BackupDir -Filter 'transcript_files_*.dump' -File)
    if ($state -ne $previous -or $existing.Count -eq 0) {
        $transcripts = Join-Path $BackupDir "transcript_files_$stamp.dump"
        Invoke-Dump $transcripts @('--data-only', '--table=public.transcript_files')
        Set-Content -LiteralPath $stateFile -Value $state
        $existing | Remove-Item -Force
        Write-Log "OK $transcripts ($state)"
    } else {
        Write-Log "UNCHANGED transcript_files ($state); kept $($existing[-1].Name)"
    }

    Write-Output $final
} catch {
    if ($partial -and (Test-Path -LiteralPath $partial)) {
        Remove-Item -LiteralPath $partial -Force
    }
    Write-Log "FAILED $($_.Exception.Message)"
    throw
}
