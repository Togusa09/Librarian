$ErrorActionPreference='Stop'
$container='librarian-librarian-1'
$port = docker inspect $container --format '{{with (index .NetworkSettings.Ports "8000/tcp")}}{{(index . 0).HostPort}}{{end}}'
if (-not $port) { throw 'Could not resolve host port for container 8000/tcp' }
$base = "http://127.0.0.1:$port"

function TimeCall([string]$name, [scriptblock]$fn){
  $sw=[Diagnostics.Stopwatch]::StartNew()
  try { $r=& $fn; $sw.Stop(); [pscustomobject]@{step=$name;status='PASS';ms=[int]$sw.ElapsedMilliseconds;data=$r} }
  catch { $sw.Stop(); [pscustomobject]@{step=$name;status='FAIL';ms=[int]$sw.ElapsedMilliseconds;error=$_.Exception.Message} }
}

$out=@()
$out += TimeCall 'healthz' { Invoke-RestMethod -Uri "$base/healthz" -TimeoutSec 20 }
$out += TimeCall 'list_files' { Invoke-RestMethod -Method Post -Uri "$base/tools/list_files" -ContentType 'application/json' -Body '{"path":""}' -TimeoutSec 30 }

$files=@()
if($out[1].status -eq 'PASS' -and $out[1].data.result.files){ $files=@($out[1].data.result.files) }
$doc = $files | Where-Object { $_ -match '\.(md|txt)$' } | Select-Object -First 1
if(-not $doc){ throw 'No .md/.txt doc found' }

$read = TimeCall 'read_document' { Invoke-RestMethod -Method Post -Uri "$base/tools/read_document" -ContentType 'application/json' -Body (@{path=$doc}|ConvertTo-Json -Compress) -TimeoutSec 60 }
$sum  = TimeCall 'summarize_context' { Invoke-RestMethod -Method Post -Uri "$base/tools/summarize_context" -ContentType 'application/json' -Body (@{scope='file';target=$doc}|ConvertTo-Json -Compress) -TimeoutSec 240 }

$readLen = 0; if($read.status -eq 'PASS' -and $read.data.result.text){ $readLen = ($read.data.result.text | Out-String).Length }
$sumLen = 0;  if($sum.status  -eq 'PASS' -and $sum.data.result.summary){ $sumLen = ($sum.data.result.summary | Out-String).Length }

[pscustomobject]@{
  container=$container
  base_url=$base
  doc_used=$doc
  read_document_ms=if($read.status -eq 'PASS'){$read.ms}else{$null}
  summarize_context_ms=if($sum.status -eq 'PASS'){$sum.ms}else{$null}
  read_document_length=$readLen
  summarize_length=$sumLen
  summarize_vs_read_ratio=if($readLen -gt 0){ [math]::Round(($sumLen/$readLen),3) } else { $null }
  steps=@($out + $read + $sum | Select-Object step,status,ms,error)
} | ConvertTo-Json -Depth 6

if($read.status -eq 'PASS'){ '--- DOC_SAMPLE ---'; ($read.data.result.text | Out-String).Substring(0,[Math]::Min(220,($read.data.result.text | Out-String).Length)) }
if($sum.status -eq 'PASS'){ '--- SUMMARY_SAMPLE ---'; ($sum.data.result.summary | Out-String).Substring(0,[Math]::Min(220,($sum.data.result.summary | Out-String).Length)) }
