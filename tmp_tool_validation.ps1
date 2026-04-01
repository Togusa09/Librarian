$ErrorActionPreference = 'Stop'
$base = 'http://127.0.0.1:8001'
function Invoke-TimedJson {
  param([string]$Name, [ScriptBlock]$Action)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  try {
    $result = & $Action
    $sw.Stop()
    [pscustomobject]@{ step=$Name; status='PASS'; ms=[int]$sw.ElapsedMilliseconds; detail='ok'; data=$result }
  } catch {
    $sw.Stop()
    $msg = $_.Exception.Message
    if ($_.ErrorDetails.Message) { $msg = $_.ErrorDetails.Message }
    [pscustomobject]@{ step=$Name; status='FAIL'; ms=[int]$sw.ElapsedMilliseconds; detail=$msg; data=$null }
  }
}
$results = @()
$results += Invoke-TimedJson 'healthz' { Invoke-RestMethod -Uri "$base/healthz" -TimeoutSec 15 }
$results += Invoke-TimedJson 'discovery' { Invoke-RestMethod -Uri "$base/.well-known/mcp.json" -TimeoutSec 15 }
$results += Invoke-TimedJson 'tools' { Invoke-RestMethod -Uri "$base/tools" -TimeoutSec 15 }
$listRes = Invoke-TimedJson 'tool:list_files' { Invoke-RestMethod -Method Post -Uri "$base/tools/list_files" -ContentType 'application/json' -Body '{"path":""}' -TimeoutSec 20 }
$results += $listRes
$files = @(); if ($listRes.status -eq 'PASS' -and $listRes.data.result.files) { $files = @($listRes.data.result.files) }
$doc = $files | Where-Object { $_ -match '\.(md|txt)$' } | Select-Object -First 1
$bin = $files | Where-Object { $_ -match '\.(png|jpg|jpeg|gif|webp|pdf|zip|bin)$' } | Select-Object -First 1
if ($doc) {
  $results += Invoke-TimedJson 'tool:read_document' { Invoke-RestMethod -Method Post -Uri "$base/tools/read_document" -ContentType 'application/json' -Body (@{path=$doc}|ConvertTo-Json -Compress) -TimeoutSec 30 }
} else { $results += [pscustomobject]@{ step='tool:read_document'; status='SKIP'; ms=0; detail='no .md/.txt file found'; data=$null } }
$results += Invoke-TimedJson 'tool:search_knowledge_base' { Invoke-RestMethod -Method Post -Uri "$base/tools/search_knowledge_base" -ContentType 'application/json' -Body '{"q":"overview architecture gameplay","top_k":3}' -TimeoutSec 45 }
if ($bin) {
  $results += Invoke-TimedJson 'tool:read_binary' { Invoke-RestMethod -Method Post -Uri "$base/tools/read_binary" -ContentType 'application/json' -Body (@{path=$bin;prefer='base64'}|ConvertTo-Json -Compress) -TimeoutSec 45 }
} else { $results += [pscustomobject]@{ step='tool:read_binary'; status='SKIP'; ms=0; detail='no binary-like file found'; data=$null } }
$summarizePass = $null
if ($doc) {
  foreach ($t in @(30,60,120,240)) {
    $attempt = Invoke-TimedJson "tool:summarize_context(timeout=${t}s)" { Invoke-RestMethod -Method Post -Uri "$base/tools/summarize_context" -ContentType 'application/json' -Body (@{scope='file';target=$doc}|ConvertTo-Json -Compress) -TimeoutSec $t }
    $results += $attempt
    if ($attempt.status -eq 'PASS') { $summarizePass = $attempt; break }
    if ($attempt.detail -notmatch 'Timeout') { break }
  }
  if (-not $summarizePass) {
    $longAttempt = Invoke-TimedJson 'tool:summarize_context(timeout=420s)' { Invoke-RestMethod -Method Post -Uri "$base/tools/summarize_context" -ContentType 'application/json' -Body (@{scope='file';target=$doc}|ConvertTo-Json -Compress) -TimeoutSec 420 }
    $results += $longAttempt
    if ($longAttempt.status -eq 'PASS') { $summarizePass = $longAttempt }
  }
}
$warmRuns = @()
if ($summarizePass) {
  1..3 | ForEach-Object {
    $wr = Invoke-TimedJson "tool:summarize_context:warm_run_$_" { Invoke-RestMethod -Method Post -Uri "$base/tools/summarize_context" -ContentType 'application/json' -Body (@{scope='file';target=$doc}|ConvertTo-Json -Compress) -TimeoutSec 180 }
    $warmRuns += $wr
    $results += $wr
  }
}
$outObj = [pscustomobject]@{ doc_used=$doc; bin_used=$bin; results=($results | Select-Object step,status,ms,detail); warm_stats=$null }
if ($warmRuns.Count -gt 0) {
  $ok = @($warmRuns | Where-Object { $_.status -eq 'PASS' })
  if ($ok.Count -gt 0) {
    $vals = @($ok | ForEach-Object { $_.ms })
    $outObj.warm_stats = [pscustomobject]@{ min_ms=($vals|Measure-Object -Minimum).Minimum; avg_ms=[int](($vals|Measure-Object -Average).Average); max_ms=($vals|Measure-Object -Maximum).Maximum }
  }
}
$outPath = 'c:\Source\Librarian\tool_validation_results.json'
$outObj | ConvertTo-Json -Depth 8 | Set-Content -Path $outPath -Encoding UTF8
Write-Output "RESULT_FILE=$outPath"
