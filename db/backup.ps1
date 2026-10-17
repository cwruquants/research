param(
    [string]$BackupDir = 'G:\pg_backups',
    [string]$PgPassFile = 'G:\pg_backups\pgpass.conf',
    [int]$RetentionDays = 14
)

$ErrorActionPreference = 'Stop'
$pgDump = 'C:\Program Files\PostgreSQL\18\bin\pg_dump.exe'
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

$stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$final = Join-Path $BackupDir "research_$stamp.dump"
$partial = "$final.partial"
$log = Join-Path $BackupDir 'backup.log'

try {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $env:PGPASSFILE = $PgPassFile
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) START user=$identity appdata=$env:APPDATA userprofile=$env:USERPROFILE pgpass_path=$PgPassFile pgpass_exists=$(Test-Path -LiteralPath $PgPassFile)"
    if (-not (Test-Path -LiteralPath $PgPassFile)) {
        try { Get-Item -LiteralPath $PgPassFile -ErrorAction Stop | Out-Null }
        catch { Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) PGPASS_ERROR $($_.Exception.Message)" }
    }
    $dumpMessages = & $pgDump -h localhost -p 5432 -U postgres -d research -w `
        --format=custom --compress=1 --file=$partial 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "pg_dump exited with code $LASTEXITCODE. $($dumpMessages.Trim())"
    }
    if ((Get-Item -LiteralPath $partial).Length -eq 0) {
        throw 'pg_dump produced an empty file'
    }
    Move-Item -LiteralPath $partial -Destination $final

    $cutoff = (Get-Date).AddDays(-$RetentionDays)
    Get-ChildItem -LiteralPath $BackupDir -Filter 'research_*.dump' -File |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        Remove-Item -Force

    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) OK $final"
    Write-Output $final
} catch {
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) FAILED $($_.Exception.Message)"
    throw
}
