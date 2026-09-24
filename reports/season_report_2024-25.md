> **RETRACTED (2026-09-24): these numbers are leak-inflated.** The model behind this
> replay was trained AND evaluated with vaastav's same-GW `xP`, which contains that GW's own
> points (see the banner in `EP_FORMULA.md`). Served the way it runs live (with honest
> pre-deadline xP), the same model scores about 1,850-2,020 on 2024-25, not 2,918 (the
> field average was 2,008). Honest, leak-free numbers: `reports/2026-27_gw1-5_review.md`.

# MILP Optimizer Season Report — 2024-25

**Strategy**: LightGBM predictions + MILP optimizer, max 1 transfer/GW
**Final Score**: 2918 net points
**Gross Points**: 2918
**Total Hits**: 0 pts (0 extra transfers)
**Total Transfers**: 37
**Chips Used**: None

## Season Summary

| GW | Formation | Gross | Hits | Net | Cumul | Captain | Transfers | Chip | Bank |
|---:|:---------:|------:|-----:|----:|------:|:--------|:----------|:-----|-----:|
| 1 | 3-5-2 | 68 | 0 | 68 | 68 | Mohamed Salah |  |  | £0.0m |
| 2 | 4-5-1 | 79 | 0 | 79 | 147 | Cole Palmer | Marcos Senesi→Cristian Romero |  | £0.0m |
| 3 | 3-5-2 | 72 | 0 | 72 | 219 | Dominic Calvert-Lewin | Dominic Solanke-Mitchell→Dominic Calvert-Lewin |  | £1.6m |
| 4 | 3-5-2 | 65 | 0 | 65 | 284 | Gabriel dos Santos Magalhães | João Cavaco Cancelo→Gabriel dos Santos Magalhães |  | £0.9m |
| 5 | 3-5-2 | 58 | 0 | 58 | 342 | Jack Hinshelwood | Carlos Henrique Casimiro→Jack Hinshelwood |  | £0.9m |
| 6 | 4-5-1 | 84 | 0 | 84 | 426 | Gabriel Martinelli Silva | Bruno Borges Fernandes→Gabriel Martinelli Silva |  | £2.3m |
| 7 | 3-5-2 | 76 | 0 | 76 | 502 | Michail Antonio | Hwang Ui-jo→Michail Antonio |  | £1.4m |
| 8 | 3-4-3 | 49 | 0 | 49 | 551 | Dejan Kulusevski | Jack Hinshelwood→Dejan Kulusevski |  | £0.2m |
| 9 | 3-4-3 | 68 | 0 | 68 | 619 | Chris Wood | Alexander Isak→Chris Wood |  | £2.2m |
| 10 | 3-4-3 | 64 | 0 | 64 | 683 | Ola Aina | Cristian Romero→Ola Aina |  | £2.6m |
| 11 | 3-5-2 | 72 | 0 | 72 | 755 | Yoane Wissa | Dominic Calvert-Lewin→Yoane Wissa |  | £2.5m |
| 12 | 3-4-3 | 89 | 0 | 89 | 844 | Aaron Wan-Bissaka | Jacob Greaves→Aaron Wan-Bissaka |  | £2.1m |
| 13 | 3-5-2 | 134 | 0 | 134 | 978 | Kevin Schade | Gabriel Martinelli Silva→Kevin Schade |  | £3.9m |
| 14 | 3-5-2 | 71 | 0 | 71 | 1049 | Jurriën Timber | Kortney Hause→Jurriën Timber |  | £2.4m |
| 15 | 3-5-2 | 89 | 0 | 89 | 1138 | Cole Palmer | Mohamed Salah→Bryan Mbeumo |  | £7.5m |
| 16 | 4-5-1 | 80 | 0 | 80 | 1218 | Ismaïla Sarr | Bryan Mbeumo→Ismaïla Sarr |  | £9.5m |
| 17 | 4-5-1 | 73 | 0 | 73 | 1291 | Ola Aina | Dejan Kulusevski→Dominik Szoboszlai |  | £9.4m |
| 18 | 3-4-3 | 70 | 0 | 70 | 1361 | Cole Palmer | Michail Antonio→Cody Gakpo |  | £7.6m |
| 19 | 3-4-3 | 68 | 0 | 68 | 1429 | Chris Wood | Bukayo Saka→Mohamed Salah |  | £4.2m |
| 20 | 3-5-2 | 95 | 0 | 95 | 1524 | Bryan Mbeumo | Dominik Szoboszlai→Bryan Mbeumo |  | £2.8m |
| 21 | 3-4-3 | 82 | 0 | 82 | 1606 | Amad Diallo | Bryan Mbeumo→Amad Diallo |  | £5.1m |
| 22 | 3-4-3 | 45 | 0 | 45 | 1651 | Chris Wood | Bernd Leno→Matz Sels |  | £5.1m |
| 23 | 3-4-3 | 103 | 0 | 103 | 1754 | Dango Ouattara | Ismaïla Sarr→Dango Ouattara |  | £5.9m |
| 24 | 3-4-3 | 131 | 0 | 131 | 1885 | Mohamed Salah | Gabriel dos Santos Magalhães→James Tarkowski |  | £7.3m |
| 25 | 3-5-2 | 92 | 0 | 92 | 1977 | Luis Díaz | Amad Diallo→Luis Díaz |  | £5.3m |
| 26 | 3-5-2 | 79 | 0 | 79 | 2056 | Marco Asensio | Cole Palmer→Marco Asensio |  | £10.1m |
| 27 | 3-5-2 | 85 | 0 | 85 | 2141 | Ismaïla Sarr | Marco Asensio→Ismaïla Sarr |  | £10.6m |
| 28 | 5-3-2 | 89 | 0 | 89 | 2230 | Marc Cucurella Saseta | Pedro Porro→Marc Cucurella Saseta |  | £10.8m |
| 29 | 5-2-3 | 56 | 0 | 56 | 2286 | Jørgen Strand Larsen | Cody Gakpo→Jørgen Strand Larsen |  | £12.9m |
| 30 | 3-5-2 | 71 | 0 | 71 | 2357 | Sandro Tonali | Dango Ouattara→Sandro Tonali |  | £12.4m |
| 31 | 3-5-2 | 59 | 0 | 59 | 2416 | Jacob Murphy | Kevin Schade→Jacob Murphy |  | £12.4m |
| 32 | 3-5-2 | 91 | 0 | 91 | 2507 | Mohamed Salah | Chris Wood→Alexander Isak |  | £9.6m |
| 33 | 3-5-2 | 80 | 0 | 80 | 2587 | Leandro Trossard | Jacob Murphy→Leandro Trossard |  | £8.0m |
| 34 | 3-4-3 | 60 | 0 | 60 | 2647 | Mohamed Salah | Ismaïla Sarr→Enzo Fernández |  | £8.9m |
| 35 | 3-5-2 | 62 | 0 | 62 | 2709 | Kevin De Bruyne | Sandro Tonali→Kevin De Bruyne |  | £5.0m |
| 36 | 3-5-2 | 64 | 0 | 64 | 2773 | Morgan Gibbs-White | Enzo Fernández→Morgan Gibbs-White |  | £3.3m |
| 37 | 3-5-2 | 85 | 0 | 85 | 2858 | Harvey Elliott | Leandro Trossard→Harvey Elliott |  | £4.8m |
| 38 | 3-5-2 | 60 | 0 | 60 | 2918 | Amad Diallo | Harvey Elliott→Amad Diallo |  | £4.7m |

---

## Gameweek Details

### GW1

**Formation**: 3-5-2 | **Free Transfers**: 1 | **Bank**: £0.0m | **Chip**: None

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Bukayo Saka | MID | 12 |  |
| 2 | Marcos Senesi | DEF | 0 | VC |
| 3 | Dominic Solanke-Mitchell | FWD | 2 |  |
| 4 | Cole Palmer | MID | 2 |  |
| 5 | Bernd Leno | GK | 3 |  |
| 6 | Mohamed Salah | MID | 14 | © Captain |
| 7 | João Cavaco Cancelo | DEF | 0 |  |
| 8 | Bruno Borges Fernandes | MID | 3 |  |
| 9 | Carlos Henrique Casimiro | MID | 3 |  |
| 10 | Alexander Isak | FWD | 5 |  |
| 11 | Pedro Porro | DEF | 9 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Hwang Ui-jo | FWD | 0 |
| 3 | Kortney Hause | DEF | 0 |
| 4 | Jacob Greaves | DEF | 1 |

**Auto-subs**:
- Marcos Senesi → Jacob Greaves

**Points**: 68 gross = **68 net** | Captain bonus: 14 | Bench pts: 0
**Cumulative**: 68

---

### GW2

**Formation**: 4-5-1 | **Free Transfers**: 2 | **Bank**: £0.0m | **Chip**: None

**Transfers**:
- OUT: Marcos Senesi → IN: Cristian Romero

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Bukayo Saka | MID | 6 | VC |
| 2 | Dominic Solanke-Mitchell | FWD | 0 |  |
| 3 | Cole Palmer | MID | 17 | © Captain |
| 4 | Bernd Leno | GK | 3 |  |
| 5 | Mohamed Salah | MID | 10 |  |
| 6 | João Cavaco Cancelo | DEF | 0 |  |
| 7 | Bruno Borges Fernandes | MID | 2 |  |
| 8 | Carlos Henrique Casimiro | MID | 2 |  |
| 9 | Alexander Isak | FWD | 2 |  |
| 10 | Pedro Porro | DEF | 6 |  |
| 11 | Cristian Romero | DEF | 14 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Hwang Ui-jo | FWD | 0 |
| 3 | Kortney Hause | DEF | 0 |
| 4 | Jacob Greaves | DEF | 0 |

**Auto-subs**:
- Dominic Solanke-Mitchell → Jacob Greaves

**Points**: 79 gross = **79 net** | Captain bonus: 17 | Bench pts: 0
**Cumulative**: 147

---

### GW3

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £0.0m | **Chip**: None

**Transfers**:
- OUT: Dominic Solanke-Mitchell → IN: Dominic Calvert-Lewin

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Bukayo Saka | MID | 6 | VC |
| 2 | Cole Palmer | MID | 6 |  |
| 3 | Bernd Leno | GK | 3 |  |
| 4 | Jacob Greaves | DEF | 2 |  |
| 5 | Mohamed Salah | MID | 17 |  |
| 6 | Bruno Borges Fernandes | MID | 2 |  |
| 7 | Carlos Henrique Casimiro | MID | 1 |  |
| 8 | Alexander Isak | FWD | 9 |  |
| 9 | Pedro Porro | DEF | 1 |  |
| 10 | Cristian Romero | DEF | 1 |  |
| 11 | Dominic Calvert-Lewin | FWD | 12 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Hwang Ui-jo | FWD | 0 |
| 3 | João Cavaco Cancelo | DEF | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Points**: 72 gross = **72 net** | Captain bonus: 12 | Bench pts: 0
**Cumulative**: 219

---

### GW4

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £1.6m | **Chip**: None

**Transfers**:
- OUT: João Cavaco Cancelo → IN: Gabriel dos Santos Magalhães

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Bukayo Saka | MID | 7 | VC |
| 2 | Cole Palmer | MID | 3 |  |
| 3 | Dominic Calvert-Lewin | FWD | 6 |  |
| 4 | Bernd Leno | GK | 2 |  |
| 5 | Jacob Greaves | DEF | 6 |  |
| 6 | Mohamed Salah | MID | 2 |  |
| 7 | Bruno Borges Fernandes | MID | 5 |  |
| 8 | Carlos Henrique Casimiro | MID | 1 |  |
| 9 | Alexander Isak | FWD | 1 |  |
| 10 | Pedro Porro | DEF | 2 |  |
| 11 | Gabriel dos Santos Magalhães | DEF | 15 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Cristian Romero | DEF | 2 |
| 3 | Hwang Ui-jo | FWD | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Points**: 65 gross = **65 net** | Captain bonus: 15 | Bench pts: 2
**Cumulative**: 284

---

### GW5

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £0.9m | **Chip**: None

**Transfers**:
- OUT: Carlos Henrique Casimiro → IN: Jack Hinshelwood

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 7 | VC |
| 2 | Bukayo Saka | MID | 4 |  |
| 3 | Cole Palmer | MID | 8 |  |
| 4 | Dominic Calvert-Lewin | FWD | 2 |  |
| 5 | Bernd Leno | GK | 2 |  |
| 6 | Jacob Greaves | DEF | 2 |  |
| 7 | Mohamed Salah | MID | 6 |  |
| 8 | Bruno Borges Fernandes | MID | 3 |  |
| 9 | Alexander Isak | FWD | 2 |  |
| 10 | Pedro Porro | DEF | 2 |  |
| 11 | Jack Hinshelwood | MID | 10 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Cristian Romero | DEF | 2 |
| 3 | Hwang Ui-jo | FWD | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Points**: 58 gross = **58 net** | Captain bonus: 10 | Bench pts: 2
**Cumulative**: 342

---

### GW6

**Formation**: 4-5-1 | **Free Transfers**: 2 | **Bank**: £0.9m | **Chip**: None

**Transfers**:
- OUT: Bruno Borges Fernandes → IN: Gabriel Martinelli Silva

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 1 | VC |
| 2 | Bukayo Saka | MID | 3 |  |
| 3 | Jack Hinshelwood | MID | 2 |  |
| 4 | Cole Palmer | MID | 25 |  |
| 5 | Dominic Calvert-Lewin | FWD | 1 |  |
| 6 | Bernd Leno | GK | 5 |  |
| 7 | Jacob Greaves | DEF | 0 |  |
| 8 | Mohamed Salah | MID | 10 |  |
| 9 | Pedro Porro | DEF | 5 |  |
| 10 | Cristian Romero | DEF | 6 |  |
| 11 | Gabriel Martinelli Silva | MID | 13 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Alexander Isak | FWD | 0 |
| 3 | Kortney Hause | DEF | 0 |
| 4 | Hwang Ui-jo | FWD | 0 |

**Points**: 84 gross = **84 net** | Captain bonus: 13 | Bench pts: 0
**Cumulative**: 426

---

### GW7

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £2.3m | **Chip**: None

**Transfers**:
- OUT: Hwang Ui-jo → IN: Michail Antonio

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 2 | VC |
| 2 | Gabriel Martinelli Silva | MID | 6 |  |
| 3 | Bukayo Saka | MID | 16 |  |
| 4 | Jack Hinshelwood | MID | 2 |  |
| 5 | Cole Palmer | MID | 6 |  |
| 6 | Dominic Calvert-Lewin | FWD | 2 |  |
| 7 | Bernd Leno | GK | 2 |  |
| 8 | Jacob Greaves | DEF | 0 |  |
| 9 | Mohamed Salah | MID | 3 |  |
| 10 | Pedro Porro | DEF | 1 |  |
| 11 | Michail Antonio | FWD | 18 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Cristian Romero | DEF | 1 |
| 3 | Alexander Isak | FWD | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Points**: 76 gross = **76 net** | Captain bonus: 18 | Bench pts: 1
**Cumulative**: 502

---

### GW8

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £1.4m | **Chip**: None

**Transfers**:
- OUT: Jack Hinshelwood → IN: Dejan Kulusevski

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 1 | VC |
| 2 | Gabriel Martinelli Silva | MID | 1 |  |
| 3 | Bukayo Saka | MID | 0 |  |
| 4 | Cole Palmer | MID | 2 |  |
| 5 | Dominic Calvert-Lewin | FWD | 2 |  |
| 6 | Bernd Leno | GK | 5 |  |
| 7 | Mohamed Salah | MID | 12 |  |
| 8 | Pedro Porro | DEF | 2 |  |
| 9 | Cristian Romero | DEF | 2 |  |
| 10 | Michail Antonio | FWD | 2 |  |
| 11 | Dejan Kulusevski | MID | 9 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Alexander Isak | FWD | 2 |
| 3 | Jacob Greaves | DEF | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Auto-subs**:
- Bukayo Saka → Alexander Isak

**Points**: 49 gross = **49 net** | Captain bonus: 9 | Bench pts: 0
**Cumulative**: 551

---

### GW9

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £0.2m | **Chip**: None

**Transfers**:
- OUT: Alexander Isak → IN: Chris Wood

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 1 | VC |
| 2 | Bukayo Saka | MID | 9 |  |
| 3 | Cole Palmer | MID | 10 |  |
| 4 | Dominic Calvert-Lewin | FWD | 2 |  |
| 5 | Bernd Leno | GK | 3 |  |
| 6 | Mohamed Salah | MID | 10 |  |
| 7 | Dejan Kulusevski | MID | 1 |  |
| 8 | Pedro Porro | DEF | 2 |  |
| 9 | Cristian Romero | DEF | 2 |  |
| 10 | Michail Antonio | FWD | 2 |  |
| 11 | Chris Wood | FWD | 13 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Gabriel Martinelli Silva | MID | 2 |
| 3 | Jacob Greaves | DEF | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Points**: 68 gross = **68 net** | Captain bonus: 13 | Bench pts: 2
**Cumulative**: 619

---

### GW10

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £2.2m | **Chip**: None

**Transfers**:
- OUT: Cristian Romero → IN: Ola Aina

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 2 | VC |
| 2 | Bukayo Saka | MID | 2 |  |
| 3 | Cole Palmer | MID | 3 |  |
| 4 | Dominic Calvert-Lewin | FWD | 2 |  |
| 5 | Bernd Leno | GK | 2 |  |
| 6 | Mohamed Salah | MID | 9 |  |
| 7 | Chris Wood | FWD | 7 |  |
| 8 | Dejan Kulusevski | MID | 5 |  |
| 9 | Pedro Porro | DEF | 1 |  |
| 10 | Michail Antonio | FWD | 1 |  |
| 11 | Ola Aina | DEF | 15 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Gabriel Martinelli Silva | MID | 2 |
| 3 | Jacob Greaves | DEF | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Points**: 64 gross = **64 net** | Captain bonus: 15 | Bench pts: 2
**Cumulative**: 683

---

### GW11

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £2.6m | **Chip**: None

**Transfers**:
- OUT: Dominic Calvert-Lewin → IN: Yoane Wissa

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 2 | VC |
| 2 | Gabriel Martinelli Silva | MID | 8 |  |
| 3 | Bukayo Saka | MID | 2 |  |
| 4 | Cole Palmer | MID | 2 |  |
| 5 | Bernd Leno | GK | 9 |  |
| 6 | Mohamed Salah | MID | 14 |  |
| 7 | Ola Aina | DEF | 1 |  |
| 8 | Chris Wood | FWD | 2 |  |
| 9 | Dejan Kulusevski | MID | 2 |  |
| 10 | Pedro Porro | DEF | 4 |  |
| 11 | Yoane Wissa | FWD | 13 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Michail Antonio | FWD | 2 |
| 3 | Jacob Greaves | DEF | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Points**: 72 gross = **72 net** | Captain bonus: 13 | Bench pts: 2
**Cumulative**: 755

---

### GW12

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £2.5m | **Chip**: None

**Transfers**:
- OUT: Jacob Greaves → IN: Aaron Wan-Bissaka

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 6 | VC |
| 2 | Gabriel Martinelli Silva | MID | 0 |  |
| 3 | Bukayo Saka | MID | 13 |  |
| 4 | Cole Palmer | MID | 2 |  |
| 5 | Bernd Leno | GK | 0 |  |
| 6 | Mohamed Salah | MID | 13 |  |
| 7 | Chris Wood | FWD | 1 |  |
| 8 | Dejan Kulusevski | MID | 6 |  |
| 9 | Pedro Porro | DEF | 14 |  |
| 10 | Michail Antonio | FWD | 2 |  |
| 11 | Aaron Wan-Bissaka | DEF | 15 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Yoane Wissa | FWD | 2 |
| 3 | Ola Aina | DEF | 0 |
| 4 | Kortney Hause | DEF | 0 |

**Auto-subs**:
- Gabriel Martinelli Silva → Yoane Wissa

**Points**: 89 gross = **89 net** | Captain bonus: 15 | Bench pts: 0
**Cumulative**: 844

---

### GW13

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £2.1m | **Chip**: None

**Transfers**:
- OUT: Gabriel Martinelli Silva → IN: Kevin Schade

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 9 | VC |
| 2 | Bukayo Saka | MID | 18 |  |
| 3 | Yoane Wissa | FWD | 8 |  |
| 4 | Cole Palmer | MID | 14 |  |
| 5 | Bernd Leno | GK | 2 |  |
| 6 | Mohamed Salah | MID | 13 |  |
| 7 | Aaron Wan-Bissaka | DEF | 6 |  |
| 8 | Ola Aina | DEF | 8 |  |
| 9 | Chris Wood | FWD | 9 |  |
| 10 | Dejan Kulusevski | MID | 1 |  |
| 11 | Kevin Schade | MID | 23 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Pedro Porro | DEF | 2 |
| 3 | Michail Antonio | FWD | 2 |
| 4 | Kortney Hause | DEF | 0 |

**Points**: 134 gross = **134 net** | Captain bonus: 23 | Bench pts: 4
**Cumulative**: 978

---

### GW14

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £3.9m | **Chip**: None

**Transfers**:
- OUT: Kortney Hause → IN: Jurriën Timber

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Bukayo Saka | MID | 3 |  |
| 2 | Kevin Schade | MID | 5 |  |
| 3 | Yoane Wissa | FWD | 2 | VC |
| 4 | Cole Palmer | MID | 7 |  |
| 5 | Bernd Leno | GK | 2 |  |
| 6 | Mohamed Salah | MID | 18 |  |
| 7 | Aaron Wan-Bissaka | DEF | 1 |  |
| 8 | Chris Wood | FWD | 2 |  |
| 9 | Dejan Kulusevski | MID | 2 |  |
| 10 | Pedro Porro | DEF | 1 |  |
| 11 | Jurriën Timber | DEF | 14 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Michail Antonio | FWD | 1 |
| 3 | Gabriel dos Santos Magalhães | DEF | 0 |
| 4 | Ola Aina | DEF | 1 |

**Points**: 71 gross = **71 net** | Captain bonus: 14 | Bench pts: 2
**Cumulative**: 1049

---

### GW15

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £2.4m | **Chip**: None

**Transfers**:
- OUT: Mohamed Salah → IN: Bryan Mbeumo

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 2 |  |
| 2 | Bukayo Saka | MID | 2 |  |
| 3 | Kevin Schade | MID | 6 |  |
| 4 | Yoane Wissa | FWD | 7 | VC |
| 5 | Cole Palmer | MID | 18 | © Captain |
| 6 | Bernd Leno | GK | 3 |  |
| 7 | Aaron Wan-Bissaka | DEF | 2 |  |
| 8 | Chris Wood | FWD | 8 |  |
| 9 | Dejan Kulusevski | MID | 7 |  |
| 10 | Pedro Porro | DEF | 3 |  |
| 11 | Bryan Mbeumo | MID | 13 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Michail Antonio | FWD | 0 |
| 3 | Gabriel dos Santos Magalhães | DEF | 0 |
| 4 | Ola Aina | DEF | 1 |

**Points**: 89 gross = **89 net** | Captain bonus: 18 | Bench pts: 1
**Cumulative**: 1138

---

### GW16

**Formation**: 4-5-1 | **Free Transfers**: 2 | **Bank**: £7.5m | **Chip**: None

**Transfers**:
- OUT: Bryan Mbeumo → IN: Ismaïla Sarr

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 9 | VC |
| 2 | Jurriën Timber | DEF | 6 |  |
| 3 | Bukayo Saka | MID | 3 |  |
| 4 | Kevin Schade | MID | 3 |  |
| 5 | Cole Palmer | MID | 3 |  |
| 6 | Bernd Leno | GK | 1 |  |
| 7 | Aaron Wan-Bissaka | DEF | 5 |  |
| 8 | Chris Wood | FWD | 2 |  |
| 9 | Dejan Kulusevski | MID | 8 |  |
| 10 | Pedro Porro | DEF | 6 |  |
| 11 | Ismaïla Sarr | MID | 17 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Ola Aina | DEF | 2 |
| 3 | Yoane Wissa | FWD | 2 |
| 4 | Michail Antonio | FWD | 0 |

**Points**: 80 gross = **80 net** | Captain bonus: 17 | Bench pts: 4
**Cumulative**: 1218

---

### GW17

**Formation**: 4-5-1 | **Free Transfers**: 2 | **Bank**: £9.5m | **Chip**: None

**Transfers**:
- OUT: Dejan Kulusevski → IN: Dominik Szoboszlai

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 4 | VC |
| 2 | Bukayo Saka | MID | 1 |  |
| 3 | Kevin Schade | MID | 2 |  |
| 4 | Cole Palmer | MID | 3 |  |
| 5 | Bernd Leno | GK | 6 |  |
| 6 | Aaron Wan-Bissaka | DEF | 2 |  |
| 7 | Ola Aina | DEF | 15 | © Captain |
| 8 | Chris Wood | FWD | 2 |  |
| 9 | Pedro Porro | DEF | -1 |  |
| 10 | Ismaïla Sarr | MID | 7 |  |
| 11 | Dominik Szoboszlai | MID | 17 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Yoane Wissa | FWD | 1 |
| 3 | Jurriën Timber | DEF | 1 |
| 4 | Michail Antonio | FWD | 0 |

**Points**: 73 gross = **73 net** | Captain bonus: 15 | Bench pts: 2
**Cumulative**: 1291

---

### GW18

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £9.4m | **Chip**: None

**Transfers**:
- OUT: Michail Antonio → IN: Cody Gakpo

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 6 | VC |
| 2 | Jurriën Timber | DEF | 9 |  |
| 3 | Kevin Schade | MID | 3 |  |
| 4 | Yoane Wissa | FWD | 2 |  |
| 5 | Cole Palmer | MID | 10 | © Captain |
| 6 | Bernd Leno | GK | 4 |  |
| 7 | Dominik Szoboszlai | MID | 0 |  |
| 8 | Aaron Wan-Bissaka | DEF | 9 |  |
| 9 | Chris Wood | FWD | 2 |  |
| 10 | Ismaïla Sarr | MID | 3 |  |
| 11 | Cody Gakpo | FWD | 12 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Ola Aina | DEF | 6 |
| 3 | Pedro Porro | DEF | 1 |
| 4 | Bukayo Saka | MID | 0 |

**Points**: 70 gross = **70 net** | Captain bonus: 10 | Bench pts: 7
**Cumulative**: 1361

---

### GW19

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £7.6m | **Chip**: None

**Transfers**:
- OUT: Bukayo Saka → IN: Mohamed Salah

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 1 |  |
| 2 | Kevin Schade | MID | 1 |  |
| 3 | Yoane Wissa | FWD | 2 | VC |
| 4 | Cole Palmer | MID | 2 |  |
| 5 | Bernd Leno | GK | 3 |  |
| 6 | Cody Gakpo | FWD | 5 |  |
| 7 | Ola Aina | DEF | 8 |  |
| 8 | Chris Wood | FWD | 12 | © Captain |
| 9 | Pedro Porro | DEF | 4 |  |
| 10 | Ismaïla Sarr | MID | 2 |  |
| 11 | Mohamed Salah | MID | 16 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Dominik Szoboszlai | MID | 0 |
| 3 | Aaron Wan-Bissaka | DEF | 0 |
| 4 | Gabriel dos Santos Magalhães | DEF | 2 |

**Points**: 68 gross = **68 net** | Captain bonus: 12 | Bench pts: 2
**Cumulative**: 1429

---

### GW20

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £4.2m | **Chip**: None

**Transfers**:
- OUT: Dominik Szoboszlai → IN: Bryan Mbeumo

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 2 | VC |
| 2 | Kevin Schade | MID | 8 |  |
| 3 | Yoane Wissa | FWD | 11 |  |
| 4 | Cole Palmer | MID | 10 |  |
| 5 | Bernd Leno | GK | 1 |  |
| 6 | Cody Gakpo | FWD | 6 |  |
| 7 | Mohamed Salah | MID | 7 |  |
| 8 | Ola Aina | DEF | 6 |  |
| 9 | Pedro Porro | DEF | 4 |  |
| 10 | Ismaïla Sarr | MID | 2 |  |
| 11 | Bryan Mbeumo | MID | 19 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Chris Wood | FWD | 8 |
| 3 | Aaron Wan-Bissaka | DEF | 0 |
| 4 | Jurriën Timber | DEF | 0 |

**Points**: 95 gross = **95 net** | Captain bonus: 19 | Bench pts: 8
**Cumulative**: 1524

---

### GW21

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £2.8m | **Chip**: None

**Transfers**:
- OUT: Bryan Mbeumo → IN: Amad Diallo

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 1 | VC |
| 2 | Yoane Wissa | FWD | 6 |  |
| 3 | Cole Palmer | MID | 10 |  |
| 4 | Bernd Leno | GK | 1 |  |
| 5 | Cody Gakpo | FWD | 2 |  |
| 6 | Mohamed Salah | MID | 2 |  |
| 7 | Aaron Wan-Bissaka | DEF | 3 |  |
| 8 | Ola Aina | DEF | 2 |  |
| 9 | Chris Wood | FWD | 9 |  |
| 10 | Ismaïla Sarr | MID | 6 |  |
| 11 | Amad Diallo | MID | 20 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Kevin Schade | MID | 1 |
| 3 | Pedro Porro | DEF | 1 |
| 4 | Jurriën Timber | DEF | 2 |

**Points**: 82 gross = **82 net** | Captain bonus: 20 | Bench pts: 4
**Cumulative**: 1606

---

### GW22

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £5.1m | **Chip**: None

**Transfers**:
- OUT: Bernd Leno → IN: Matz Sels

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 1 | VC |
| 2 | Yoane Wissa | FWD | 2 |  |
| 3 | Cole Palmer | MID | 2 |  |
| 4 | Cody Gakpo | FWD | 2 |  |
| 5 | Mohamed Salah | MID | 3 |  |
| 6 | Amad Diallo | MID | 2 |  |
| 7 | Aaron Wan-Bissaka | DEF | 1 |  |
| 8 | Ola Aina | DEF | 4 |  |
| 9 | Chris Wood | FWD | 12 | © Captain |
| 10 | Ismaïla Sarr | MID | 3 |  |
| 11 | Matz Sels | GK | 1 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Kevin Schade | MID | 1 |
| 3 | Pedro Porro | DEF | 1 |
| 4 | Jurriën Timber | DEF | 1 |

**Points**: 45 gross = **45 net** | Captain bonus: 12 | Bench pts: 3
**Cumulative**: 1651

---

### GW23

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £5.1m | **Chip**: None

**Transfers**:
- OUT: Ismaïla Sarr → IN: Dango Ouattara

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Gabriel dos Santos Magalhães | DEF | 7 | VC |
| 2 | Jurriën Timber | DEF | 5 |  |
| 3 | Kevin Schade | MID | 11 |  |
| 4 | Yoane Wissa | FWD | 2 |  |
| 5 | Cody Gakpo | FWD | 16 |  |
| 6 | Mohamed Salah | MID | 8 |  |
| 7 | Amad Diallo | MID | 3 |  |
| 8 | Matz Sels | GK | 1 |  |
| 9 | Chris Wood | FWD | 2 |  |
| 10 | Pedro Porro | DEF | 6 |  |
| 11 | Dango Ouattara | MID | 21 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Cole Palmer | MID | 2 |
| 3 | Ola Aina | DEF | 1 |
| 4 | Aaron Wan-Bissaka | DEF | 2 |

**Points**: 103 gross = **103 net** | Captain bonus: 21 | Bench pts: 5
**Cumulative**: 1754

---

### GW24

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £5.9m | **Chip**: None

**Transfers**:
- OUT: Gabriel dos Santos Magalhães → IN: James Tarkowski

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Dango Ouattara | MID | 2 |  |
| 2 | Yoane Wissa | FWD | 2 | VC |
| 3 | Cole Palmer | MID | 6 |  |
| 4 | Cody Gakpo | FWD | 7 |  |
| 5 | Mohamed Salah | MID | 29 | © Captain |
| 6 | Amad Diallo | MID | 2 |  |
| 7 | Ola Aina | DEF | 6 |  |
| 8 | Matz Sels | GK | 8 |  |
| 9 | Chris Wood | FWD | 17 |  |
| 10 | Pedro Porro | DEF | 6 |  |
| 11 | James Tarkowski | DEF | 17 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Kevin Schade | MID | 2 |
| 3 | Jurriën Timber | DEF | 1 |
| 4 | Aaron Wan-Bissaka | DEF | -1 |

**Points**: 131 gross = **131 net** | Captain bonus: 29 | Bench pts: 2
**Cumulative**: 1885

---

### GW25

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £7.3m | **Chip**: None

**Transfers**:
- OUT: Amad Diallo → IN: Luis Díaz

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 6 | VC |
| 2 | Dango Ouattara | MID | 7 |  |
| 3 | Kevin Schade | MID | 9 |  |
| 4 | Yoane Wissa | FWD | 2 |  |
| 5 | James Tarkowski | DEF | 2 |  |
| 6 | Cody Gakpo | FWD | 0 |  |
| 7 | Mohamed Salah | MID | 20 |  |
| 8 | Matz Sels | GK | 3 |  |
| 9 | Chris Wood | FWD | 9 |  |
| 10 | Pedro Porro | DEF | 6 |  |
| 11 | Luis Díaz | MID | 13 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Cole Palmer | MID | 2 |
| 3 | Ola Aina | DEF | 1 |
| 4 | Aaron Wan-Bissaka | DEF | 2 |

**Auto-subs**:
- Cody Gakpo → Cole Palmer

**Points**: 92 gross = **92 net** | Captain bonus: 13 | Bench pts: 3
**Cumulative**: 1977

---

### GW26

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £5.3m | **Chip**: None

**Transfers**:
- OUT: Cole Palmer → IN: Marco Asensio

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Dango Ouattara | MID | 2 |  |
| 2 | Kevin Schade | MID | 3 |  |
| 3 | Yoane Wissa | FWD | 8 |  |
| 4 | James Tarkowski | DEF | 1 |  |
| 5 | Luis Díaz | MID | 3 |  |
| 6 | Mohamed Salah | MID | 14 | VC |
| 7 | Aaron Wan-Bissaka | DEF | 12 |  |
| 8 | Matz Sels | GK | 0 |  |
| 9 | Chris Wood | FWD | 5 |  |
| 10 | Pedro Porro | DEF | 1 |  |
| 11 | Marco Asensio | MID | 15 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Jurriën Timber | DEF | 2 |
| 3 | Cody Gakpo | FWD | 1 |
| 4 | Ola Aina | DEF | 0 |

**Points**: 79 gross = **79 net** | Captain bonus: 15 | Bench pts: 3
**Cumulative**: 2056

---

### GW27

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £10.1m | **Chip**: None

**Transfers**:
- OUT: Marco Asensio → IN: Ismaïla Sarr

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 6 | VC |
| 2 | Dango Ouattara | MID | 2 |  |
| 3 | Kevin Schade | MID | 3 |  |
| 4 | Yoane Wissa | FWD | 9 |  |
| 5 | Luis Díaz | MID | 8 |  |
| 6 | Mohamed Salah | MID | 6 |  |
| 7 | Aaron Wan-Bissaka | DEF | 7 |  |
| 8 | Ola Aina | DEF | 6 |  |
| 9 | Matz Sels | GK | 6 |  |
| 10 | Chris Wood | FWD | 2 |  |
| 11 | Ismaïla Sarr | MID | 15 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Pedro Porro | DEF | 2 |
| 3 | Cody Gakpo | FWD | 1 |
| 4 | James Tarkowski | DEF | 2 |

**Points**: 85 gross = **85 net** | Captain bonus: 15 | Bench pts: 5
**Cumulative**: 2141

---

### GW28

**Formation**: 5-3-2 | **Free Transfers**: 2 | **Bank**: £10.6m | **Chip**: None

**Transfers**:
- OUT: Pedro Porro → IN: Marc Cucurella Saseta

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 5 | VC |
| 2 | Yoane Wissa | FWD | 2 |  |
| 3 | James Tarkowski | DEF | 2 |  |
| 4 | Luis Díaz | MID | 6 |  |
| 5 | Mohamed Salah | MID | 15 |  |
| 6 | Aaron Wan-Bissaka | DEF | 2 |  |
| 7 | Ola Aina | DEF | 6 |  |
| 8 | Matz Sels | GK | 8 |  |
| 9 | Chris Wood | FWD | 2 |  |
| 10 | Ismaïla Sarr | MID | 11 |  |
| 11 | Marc Cucurella Saseta | DEF | 15 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Kevin Schade | MID | 2 |
| 3 | Cody Gakpo | FWD | 0 |
| 4 | Dango Ouattara | MID | 1 |

**Points**: 89 gross = **89 net** | Captain bonus: 15 | Bench pts: 3
**Cumulative**: 2230

---

### GW29

**Formation**: 5-2-3 | **Free Transfers**: 2 | **Bank**: £10.8m | **Chip**: None

**Transfers**:
- OUT: Cody Gakpo → IN: Jørgen Strand Larsen

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 9 | VC |
| 2 | Dango Ouattara | MID | 1 |  |
| 3 | Kevin Schade | MID | 2 |  |
| 4 | Yoane Wissa | FWD | 8 |  |
| 5 | Marc Cucurella Saseta | DEF | 2 |  |
| 6 | James Tarkowski | DEF | 2 |  |
| 7 | Aaron Wan-Bissaka | DEF | 2 |  |
| 8 | Ola Aina | DEF | 1 |  |
| 9 | Matz Sels | GK | 1 |  |
| 10 | Chris Wood | FWD | 2 |  |
| 11 | Jørgen Strand Larsen | FWD | 13 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Mohamed Salah | MID | 0 |
| 3 | Ismaïla Sarr | MID | 0 |
| 4 | Luis Díaz | MID | 0 |

**Points**: 56 gross = **56 net** | Captain bonus: 13 | Bench pts: 0
**Cumulative**: 2286

---

### GW30

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £12.9m | **Chip**: None

**Transfers**:
- OUT: Dango Ouattara → IN: Sandro Tonali

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 7 | VC |
| 2 | Kevin Schade | MID | 2 |  |
| 3 | Marc Cucurella Saseta | DEF | 5 |  |
| 4 | Luis Díaz | MID | 9 |  |
| 5 | Mohamed Salah | MID | 3 |  |
| 6 | Ola Aina | DEF | 1 |  |
| 7 | Matz Sels | GK | 8 |  |
| 8 | Chris Wood | FWD | 0 |  |
| 9 | Jørgen Strand Larsen | FWD | 9 |  |
| 10 | Ismaïla Sarr | MID | 2 |  |
| 11 | Sandro Tonali | MID | 10 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Yoane Wissa | FWD | 5 |
| 3 | Aaron Wan-Bissaka | DEF | 2 |
| 4 | James Tarkowski | DEF | 1 |

**Auto-subs**:
- Chris Wood → Yoane Wissa

**Points**: 71 gross = **71 net** | Captain bonus: 10 | Bench pts: 3
**Cumulative**: 2357

---

### GW31

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £12.4m | **Chip**: None

**Transfers**:
- OUT: Kevin Schade → IN: Jacob Murphy

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 1 | VC |
| 2 | Yoane Wissa | FWD | 2 |  |
| 3 | James Tarkowski | DEF | 1 |  |
| 4 | Luis Díaz | MID | 6 |  |
| 5 | Mohamed Salah | MID | 2 |  |
| 6 | Aaron Wan-Bissaka | DEF | 1 |  |
| 7 | Sandro Tonali | MID | 3 |  |
| 8 | Matz Sels | GK | 3 |  |
| 9 | Jørgen Strand Larsen | FWD | 6 |  |
| 10 | Ismaïla Sarr | MID | 2 |  |
| 11 | Jacob Murphy | MID | 16 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Chris Wood | FWD | 0 |
| 3 | Marc Cucurella Saseta | DEF | 1 |
| 4 | Ola Aina | DEF | 0 |

**Points**: 59 gross = **59 net** | Captain bonus: 16 | Bench pts: 1
**Cumulative**: 2416

---

### GW32

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £12.4m | **Chip**: None

**Transfers**:
- OUT: Chris Wood → IN: Alexander Isak

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Marc Cucurella Saseta | DEF | 1 |  |
| 2 | James Tarkowski | DEF | 7 |  |
| 3 | Luis Díaz | MID | 9 |  |
| 4 | Mohamed Salah | MID | 6 | © Captain |
| 5 | Aaron Wan-Bissaka | DEF | 1 |  |
| 6 | Jacob Murphy | MID | 19 |  |
| 7 | Sandro Tonali | MID | 15 | VC |
| 8 | Matz Sels | GK | 3 |  |
| 9 | Jørgen Strand Larsen | FWD | 6 |  |
| 10 | Ismaïla Sarr | MID | 7 |  |
| 11 | Alexander Isak | FWD | 11 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Yoane Wissa | FWD | 9 |
| 3 | Jurriën Timber | DEF | 1 |
| 4 | Ola Aina | DEF | 0 |

**Points**: 91 gross = **91 net** | Captain bonus: 6 | Bench pts: 10
**Cumulative**: 2507

---

### GW33

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £9.6m | **Chip**: None

**Transfers**:
- OUT: Jacob Murphy → IN: Leandro Trossard

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 4 | VC |
| 2 | Yoane Wissa | FWD | 11 |  |
| 3 | James Tarkowski | DEF | 1 |  |
| 4 | Luis Díaz | MID | 3 |  |
| 5 | Mohamed Salah | MID | 3 |  |
| 6 | Aaron Wan-Bissaka | DEF | 0 |  |
| 7 | Sandro Tonali | MID | 2 |  |
| 8 | Matz Sels | GK | 2 |  |
| 9 | Jørgen Strand Larsen | FWD | 2 |  |
| 10 | Ismaïla Sarr | MID | 3 |  |
| 11 | Leandro Trossard | MID | 24 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Alexander Isak | FWD | 2 |
| 3 | Marc Cucurella Saseta | DEF | 1 |
| 4 | Ola Aina | DEF | 0 |

**Auto-subs**:
- Aaron Wan-Bissaka → Marc Cucurella Saseta

**Points**: 80 gross = **80 net** | Captain bonus: 24 | Bench pts: 2
**Cumulative**: 2587

---

### GW34

**Formation**: 3-4-3 | **Free Transfers**: 2 | **Bank**: £8.0m | **Chip**: None

**Transfers**:
- OUT: Ismaïla Sarr → IN: Enzo Fernández

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Yoane Wissa | FWD | 7 | VC |
| 2 | Marc Cucurella Saseta | DEF | 6 |  |
| 3 | James Tarkowski | DEF | 0 |  |
| 4 | Luis Díaz | MID | 7 |  |
| 5 | Mohamed Salah | MID | 8 | © Captain |
| 6 | Aaron Wan-Bissaka | DEF | 1 |  |
| 7 | Alexander Isak | FWD | 6 |  |
| 8 | Sandro Tonali | MID | 3 |  |
| 9 | Matz Sels | GK | 1 |  |
| 10 | Jørgen Strand Larsen | FWD | 6 |  |
| 11 | Enzo Fernández | MID | 6 |  |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Ola Aina | DEF | 1 |
| 3 | Jurriën Timber | DEF | 0 |
| 4 | Leandro Trossard | MID | 0 |

**Auto-subs**:
- James Tarkowski → Ola Aina

**Points**: 60 gross = **60 net** | Captain bonus: 8 | Bench pts: 0
**Cumulative**: 2647

---

### GW35

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £8.9m | **Chip**: None

**Transfers**:
- OUT: Sandro Tonali → IN: Kevin De Bruyne

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Jurriën Timber | DEF | 0 | VC |
| 2 | Leandro Trossard | MID | 2 |  |
| 3 | Yoane Wissa | FWD | 7 |  |
| 4 | Enzo Fernández | MID | 9 |  |
| 5 | James Tarkowski | DEF | 0 |  |
| 6 | Luis Díaz | MID | 0 |  |
| 7 | Mohamed Salah | MID | 2 |  |
| 8 | Aaron Wan-Bissaka | DEF | 7 |  |
| 9 | Alexander Isak | FWD | 9 |  |
| 10 | Matz Sels | GK | 1 |  |
| 11 | Kevin De Bruyne | MID | 11 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Jørgen Strand Larsen | FWD | 0 |
| 3 | Marc Cucurella Saseta | DEF | 2 |
| 4 | Ola Aina | DEF | 1 |

**Auto-subs**:
- Jurriën Timber → Marc Cucurella Saseta
- James Tarkowski → Ola Aina

**Points**: 62 gross = **62 net** | Captain bonus: 11 | Bench pts: 0
**Cumulative**: 2709

---

### GW36

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £5.0m | **Chip**: None

**Transfers**:
- OUT: Enzo Fernández → IN: Morgan Gibbs-White

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Leandro Trossard | MID | 7 |  |
| 2 | Yoane Wissa | FWD | 1 |  |
| 3 | Marc Cucurella Saseta | DEF | 1 |  |
| 4 | Luis Díaz | MID | 8 |  |
| 5 | Mohamed Salah | MID | 2 |  |
| 6 | Kevin De Bruyne | MID | 3 | VC |
| 7 | Aaron Wan-Bissaka | DEF | 12 |  |
| 8 | Alexander Isak | FWD | 2 |  |
| 9 | Ola Aina | DEF | 0 |  |
| 10 | Matz Sels | GK | 2 |  |
| 11 | Morgan Gibbs-White | MID | 13 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Jørgen Strand Larsen | FWD | 1 |
| 3 | Jurriën Timber | DEF | 0 |
| 4 | James Tarkowski | DEF | 0 |

**Points**: 64 gross = **64 net** | Captain bonus: 13 | Bench pts: 1
**Cumulative**: 2773

---

### GW37

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £3.3m | **Chip**: None

**Transfers**:
- OUT: Leandro Trossard → IN: Harvey Elliott

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Yoane Wissa | FWD | 10 |  |
| 2 | Marc Cucurella Saseta | DEF | 15 |  |
| 3 | Luis Díaz | MID | 1 |  |
| 4 | Mohamed Salah | MID | 2 |  |
| 5 | Kevin De Bruyne | MID | 3 |  |
| 6 | Aaron Wan-Bissaka | DEF | 1 | VC |
| 7 | Ola Aina | DEF | 2 |  |
| 8 | Morgan Gibbs-White | MID | 11 |  |
| 9 | Matz Sels | GK | 3 |  |
| 10 | Jørgen Strand Larsen | FWD | 11 |  |
| 11 | Harvey Elliott | MID | 13 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Alexander Isak | FWD | 0 |
| 3 | Jurriën Timber | DEF | 0 |
| 4 | James Tarkowski | DEF | 0 |

**Points**: 85 gross = **85 net** | Captain bonus: 13 | Bench pts: 0
**Cumulative**: 2858

---

### GW38

**Formation**: 3-5-2 | **Free Transfers**: 2 | **Bank**: £4.8m | **Chip**: None

**Transfers**:
- OUT: Harvey Elliott → IN: Amad Diallo

**Starting XI**:

| # | Player | Pos | Pts | Role |
|--:|:-------|:---:|----:|:-----|
| 1 | Yoane Wissa | FWD | 1 |  |
| 2 | Marc Cucurella Saseta | DEF | 8 |  |
| 3 | Luis Díaz | MID | 2 |  |
| 4 | Mohamed Salah | MID | 10 |  |
| 5 | Kevin De Bruyne | MID | 1 |  |
| 6 | Aaron Wan-Bissaka | DEF | 5 | VC |
| 7 | Alexander Isak | FWD | 2 |  |
| 8 | Ola Aina | DEF | 1 |  |
| 9 | Morgan Gibbs-White | MID | 2 |  |
| 10 | Matz Sels | GK | 2 |  |
| 11 | Amad Diallo | MID | 13 | © Captain |

**Bench**:

| # | Player | Pos | Pts |
|--:|:-------|:---:|----:|
| 1 | Cieran Slicker | GK | 0 |
| 2 | Jørgen Strand Larsen | FWD | 4 |
| 3 | James Tarkowski | DEF | 0 |
| 4 | Jurriën Timber | DEF | 0 |

**Points**: 60 gross = **60 net** | Captain bonus: 13 | Bench pts: 4
**Cumulative**: 2918

---
