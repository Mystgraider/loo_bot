# Lotto Statistical Analyzer + Daily Telegram Report

Automated na tumatakbo araw-araw via **GitHub Actions** (kaya walang PC/laptop
na kailangan — Android + browser lang ang gagamitin mo) at nagpapadala ng
report sa Telegram.

## ⚠️ Basahin muna ito

Ang script na ito ay gumagawa ng **statistical exploration** lang (frequency
count, Poisson fairness test, at XGBoost pattern-scoring) sa historical
draws. **Independent random event ang bawat lotto draw** — walang statistical
method o ML model na kayang tumpakan nang tama ang eksaktong 6-number
combination ng susunod na draw. Ang "recommendation" na ipapadala nito ay
**hindi guarantee ng panalo**. Gamitin ito bilang libangan/research tool,
hindi bilang batayan ng malaking bet.

## Ano ang laman

```
lotto-predictor/
├── config.py                  # listahan ng mga games (6/55, 6/58, ez2, atbp.)
├── analyzer.py                 # frequency + Poisson + ML logic
├── predict.py                  # main script — tumatakbo per game, gumagawa ng results.json
├── telegram_notify.py          # nagpapadala ng results.json sa Telegram
├── requirements.txt
├── data/
│   ├── draws_6_55.csv          # SAMPLE DATA LANG — palitan ng totoong resulta
│   ├── draws_ez2.csv
│   └── draws_swertres.csv
└── .github/workflows/
    └── daily-predict.yml       # cron job — tumatakbo araw-araw
```

## 🔴 IMPORTANTENG PAALALA — ngayon AUTOMATIC na, pero third-party source

Idinagdag na ang `scraper.py` na kumukuha ng pinaka-bagong resulta mula sa
**lottopcso.com** (isang independent/unofficial na site na regular
na-a-update, hindi opisyal na PCSO source dahil walang libreng public API
ang PCSO mismo). Bahagi na ito ng daily workflow: **scrape → i-commit ang
bagong data → analyze → send sa Telegram** — lahat automatic, walang
kailangang gawin manually.

Dahil ito ay pag-scrape sa isang third-party site (hindi opisyal na API):
- Puwedeng magbago ang HTML structure ng site anumang oras, at kapag
  nangyari 'yon, puwedeng mag-warning o tumigil ang scraper sa isang laro
  (hindi naman sa lahat — nakahiwalay ang error handling per laro).
- **I-check ang unang ilang GitHub Actions run** para makumpirma na
  gumagana ang scraper bago mo ito lubusang asahan. Puntahan ang tab na
  **Actions** sa repo mo pagkatapos ng unang scheduled run (o i-trigger
  manually via "Run workflow") at tingnan ang logs.
- Kung mag-warn ang scraper ("hindi mahanap ang history table") para sa
  isang laro, manual mo na lang muna idagdag ang resulta sa CSV habang
  hinihintay ang ayos.
- Palagi ka pa ring puwedeng manual na mag-edit ng CSV kahit gumagana na
  ang scraper — hindi mag-o-overwrite ang scraper sa mga existing na row,
  dagdag lang ito ng bagong petsa/draw.

## Setup (gagawin mo isang beses lang)

### 1. Gumawa ng Telegram bot
1. Sa Telegram, hanapin si **@BotFather**, i-message ng `/newbot`.
2. Sundin ang instructions, kunin ang **bot token** (parang
   `123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
3. I-message ang bagong bot mo ng kahit ano (para ma-activate), tapos i-open sa
   browser: `https://api.telegram.org/bot<TOKEN>/getUpdates` — hahanapin mo
   ang `"chat":{"id": ...}` doon. 'Yon ang **chat ID** mo.

### 2. Gumawa ng GitHub repo
1. Sa GitHub app o mobile browser, gumawa ng **bagong PRIVATE repo**
   (private para hindi makita ng iba ang data mo).
2. I-upload ang lahat ng files dito (drag-drop sa "Add file → Upload files"
   sa GitHub web UI — gumagana ito sa mobile browser).

### 3. Idagdag ang mga secrets
Sa repo mo: **Settings → Secrets and variables → Actions → New repository
secret**
- `TELEGRAM_BOT_TOKEN` = yung token mula sa BotFather
- `TELEGRAM_CHAT_ID` = yung chat ID na nakuha mo

### 4. I-enable ang Actions
Pumunta sa tab na **Actions** ng repo mo → i-click ang workflow na
"Daily Lotto Analysis" → **Enable workflow** (kung naka-disable pa).

Puwede mo ring i-test agad: **Run workflow** button (manual trigger) para
makita agad kung gumagana bago pa dumating ang scheduled time.

### 5. Real na PCSO data + automatic na updates
Kasama na ang **totoong PCSO draw results** (Jan 31 – Aug 2, 2026) bilang
starting data sa `data/*.csv`, kasama na ang 9 laro: 6/55, 6/58, 6/49,
6/45, 6/42, EZ2 (2D), Swertres (3D), 6D, at 4D. Mula rito, awtomatiko nang
kukuha ng bagong resulta ang `scraper.py` bawat araw bago tumakbo ang
analysis -- tingnan ang paalala sa itaas.

## Paano baguhin ang oras ng pagpapadala
Sa `.github/workflows/daily-predict.yml`, i-edit ang cron line. Ang GitHub
Actions ay gumagamit ng **UTC time**. Halimbawa, para sa 7:00 AM Manila time
(UTC+8): `cron: "0 23 * * *"` (11PM UTC ng nakaraang araw).

## Pagdaragdag ng ibang laro
Buksan ang `config.py`, magdagdag ng entry sa `GAMES` dict (halimbawa ng
6D o ibang bagong laro ng PCSO), tapos gumawa ng kaukulang CSV sa `data/`.
Automatic na masasama ito sa susunod na run at sa Telegram report.

## Bakit hindi ito "totoong predictor"
Basahin ang buong paliwanag na ibinigay ko sa chat — buod: `C(55,6) =
28,989,675` posibleng combinations sa 6/55, independent ang bawat draw
(walang "memory" mula sa nakaraang resulta), at kahit anong pattern na
makikita ng ML model sa 500 draws ay malamang noise/overfitting lang, hindi
totoong signal. Ang Poisson test dito ay para lang tingnan kung may
statistical bias ang machine — hindi ito nagbibigay ng edge sa
pag-predict ng eksaktong numero.
