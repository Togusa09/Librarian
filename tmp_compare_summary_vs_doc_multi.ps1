$ErrorActionPreference='Stop'
$container='librarian-librarian-1'
$port = docker inspect $container --format '{{with (index .NetworkSettings.Ports "8000/tcp")}}{{(index . 0).HostPort}}{{end}}'
$base = "http://127.0.0.1:$port"

function TimeCall([scriptblock]$fn){
  $sw=[Diagnostics.Stopwatch]::StartNew()
  $r=& $fn
  $sw.Stop()
  [pscustomobject]@{ms=[int]$sw.ElapsedMilliseconds;data=$r}
}

$list = Invoke-RestMethod -Method Post -Uri "$base/tools/list_files" -ContentType 'application/json' -Body '{"path":""}' -TimeoutSec 30
$docs = @($list.result.files | Where-Object { $_ -match '\.(md|txt)$' } | Select-Object -First 5)
if($docs.Count -eq 0){ throw 'No docs found' }

$rows=@()
foreach($d in $docs){
  $read = TimeCall { Invoke-RestMethod -Method Post -Uri "$base/tools/read_document" -ContentType 'application/json' -Body (@{path=$d}|ConvertTo-Json -Compress) -TimeoutSec 60 }
  $sum  = TimeCall { Invoke-RestMethod -Method Post -Uri "$base/tools/summarize_context" -ContentType 'application/json' -Body (@{scope='file';target=$d}|ConvertTo-Json -Compress) -TimeoutSec 240 }
  $readLen = 0; if($read.data.result.text){ $readLen = ($read.data.result.text|Out-String).Length }
  $sumLen  = 0; if($sum.data.result.summary){ $sumLen = ($sum.data.result.summary|Out-String).Length }
  $rows += [pscustomobject]@{
    doc=$d
    read_ms=$read.ms
    summarize_ms=$sum.ms
    read_len=$readLen
    summary_len=$sumLen
    ratio=if($readLen -gt 0){ [math]::Round(($sumLen/$readLen),3) } else { $null }
  }
}

$agg = [pscustomobject]@{
  docs_tested=$rows.Count
  avg_read_ms=[int](($rows.read_ms | Measure-Object -Average).Average)
  avg_summarize_ms=[int](($rows.summarize_ms | Measure-Object -Average).Average)
  avg_read_len=[int](($rows.read_len | Measure-Object -Average).Average)
  avg_summary_len=[int](($rows.summary_len | Measure-Object -Average).Average)
}

[pscustomobject]@{container=$container;base_url=$base;aggregate=$agg;rows=$rows} | ConvertTo-Json -Depth 6
