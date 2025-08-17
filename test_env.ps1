# 1) Proje köküne geç
Set-Location C:\Project_Chimera

# 2) Varsa bozuk .env'yi kaldır
if (Test-Path .env) { Remove-Item -Force .env }

# 3) İçeriği tanımla (anahtarlarını buraya yazıyorum)
$envText = @"
PEXELS_API_KEY=SkEG6SqXRKE6OzoUVqlATFTn9hC8jmrf7TRimoA9D6wt8ME9ZCirpscf
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b
"@

# 4) .env dosyasını UTF-8 (BOM YOK) formatında oluştur
[System.IO.File]::WriteAllText(".env", $envText, [System.Text.UTF8Encoding]::new($false))

# 5) İçeriği kontrol et (görüntülenmeli)
Get-Content .env

# 6) Python’dan yüklenebildiğini hızlı test et (hata vermemeli, model adını yazdırmalı)
python - <<'PY'
from dotenv import load_dotenv
import os
load_dotenv()
print("OLLAMA_MODEL =", os.getenv("OLLAMA_MODEL"))
PY
