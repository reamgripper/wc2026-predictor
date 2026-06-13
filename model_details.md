;; ============================================================================
;; MODEL DETAILS — editable content for the "Model Details" page.
;;
;; HOW TO EDIT:
;;   * Just change the text. Most of this file is plain Markdown
;;     (## headings, **bold**, tables, - bullet lists).
;;   * Lines starting with ;; are comments and are ignored.
;;
;; SPECIAL DIRECTIVES (each on its own line):
;;   @@TITLE: Big heading | smaller subtitle text
;;   @@CARD@@                         start a new rounded "card" panel
;;   @@BADGE: Step 01@@               little pill label at the top of a card
;;   @@PILLS: Tag A | Tag B | Tag C@@ row of small rounded tags
;;   @@FLOW: Box A | Box B | Box C@@  arrowed flow diagram
;;   @@LATEX@@ ... @@END@@            a centred math formula (LaTeX)
;;   @@CODE@@ ... @@END@@             a monospace code block
;;   @@NOTE:green@@ ... @@END@@       green callout box  (also: blue, amber)
;;   @@GLOSSARY@@ ... @@END@@         collapsible "Key terms" list;
;;                                    one term per line as:  Term :: definition
;; ============================================================================

@@TITLE: How the Model Works | FIFA World Cup 2026 - my prediction engine - the full mathematical walkthrough, written so someone learning statistics can follow exactly what I built

### My pipeline at a glance
I built the predictor as a clear chain of steps. Each section below explains one box.

@@FLOW: Raw match data | Feature engineering | Poisson GLM -> lambda | Form & injury adjustments | Market blend | Monte Carlo x 10,000 | Win% / CI / Scoreline

@@PILLS: Poisson GLM | Dixon-Coles | ELO ratings | WC Heritage (jfjelstul) | Tournament form lambda-adjustment | Injury penalties | DraftKings blend | Polymarket | Monte Carlo | Skellam CI

@@GLOSSARY@@
lambda - expected goals :: The average number of goals a team is expected to score. lambda = 1.8 means about 1.8 goals on average. It is not a guarantee -- it is the long-run average over many identical match-ups. Every other number I report flows from lambda.
Poisson distribution :: The standard probability distribution for counting rare, independent events in a fixed time window -- like goals in 90 minutes. Given lambda, it tells you P(0 goals), P(1 goal), P(2 goals), ...
Generalised Linear Model (GLM) :: A regression technique that learns, from thousands of past matches, how inputs (team form, ELO, venue) translate to an output. Here the output is count data (goals), so I use a Poisson GLM with a log link function.
Log link function :: Because goals must be >= 0, the GLM works on log(lambda) -- a number that can be any positive or negative value -- rather than lambda directly. This prevents the model from ever predicting negative goals.
ELO rating :: A single number measuring a team overall strength. Spain is approximately 2150; a new qualifier approximately 1300. The gap between two teams ELOs predicts match outcome even without recent data.
Dixon-Coles correction :: A small tweak that increases the modelled probability of 0-0, 1-0, 0-1, and 1-1 results -- correcting the known fact that Poisson over-predicts exactly those mid-low scorelines.
Monte Carlo simulation :: Repeating a random experiment thousands of times to estimate probabilities. Named after the casino. Each run generates one random scoreline; 10,000 runs reveal the full distribution of possible outcomes.
Confidence interval (CI) :: The range likely to contain the true result. A 95% CI of [-1, +3] means 95% of simulated games landed between Team A losing by 1 and Team A winning by 3.
Skellam distribution :: The exact probability distribution of the difference between two independent Poisson random variables (i.e. goal differential). I use it to compute the CI analytically rather than only from the MC sample.
Vig / overround :: The bookmaker profit margin. Raw moneyline probabilities add up to more than 100%. De-vigging rescales them to sum to exactly 100%, recovering the fair probabilities the market actually implies.
Tournament form adjustment :: A multiplicative correction applied to lambda after the GLM, based on how a team is actually performing in the current tournament. Ramps up from zero influence (0 games played) to a maximum 35% influence (3+ games played).
WC Heritage features (jfjelstul) :: Per-team statistics derived from every World Cup match 1930-2022: WC-specific goals/game, conceded/game, and average stage reached. I feed these into the GLM as covariates so it can distinguish between a team that is generally strong (high ELO) and one with proven WC pedigree.
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 01@@
### Data sources
I feed my model from five free public sources -- no API key required, and I cache them all locally so the app starts fast on repeat visits.

| Source | Data | Used for |
|---|---|---|
| **Kaggle -- martj42** | 47k international matches 1872-2026 | Probably way too much Historical training data, but it will be fixed by giving more weightage to recent matches.  |
| **jfjelstul/worldcup** (GitHub) | 1,248 WC matches 1930-2022, 20+ tables | WC-specific heritage features fed directly into the GLM |
| **eloratings.net** | Live strength ratings for 240+ nations | ELO covariate; refreshed every 24 h |
| **ESPN** (unofficial) | WC 2026 fixtures / results / squads / per-player stats / DraftKings odds | Live 2026 context: tournament form, injuries, betting blend |
| **Polymarket** | Prediction market "Will X win WC 2026?" prices | Crowd-intelligence tournament-winner probabilities |

@@NOTE:green@@
**Why I use multiple sources:** the Kaggle dataset gives me breadth (decades of form); jfjelstul gives me depth on World Cup behaviour specifically; ELO gives me an independent quality anchor; ESPN tells me what is happening right now; and Polymarket gives me crowd wisdom on tournament trajectory.
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 02@@
### Feature engineering -- turning matches into numbers
Raw match records are useless to me directly. I transform them into features -- numbers that encode why a team is likely to score more or fewer goals.

#### 2a - Time-weighted rolling form
For each team, I look at its last 10 matches before the date being evaluated. I let recent games carry more weight than old ones via an exponential decay factor of 0.85.

@@LATEX@@
w_i = \frac{0.85^{\,(n-1-i)}}{\sum_{j=0}^{n-1} 0.85^{\,j}}
\qquad \text{for } i = 0, 1, \ldots, n-1
@@END@@

where i=0 is the oldest match and i=n-1 is the most recent. The weights sum to 1, so this is a proper weighted average.

@@NOTE:blue@@
**Example:** with 3 matches the weights are approximately 0.30, 0.35, 0.41 -- the most recent game counts 41%, the oldest only 30%.
@@END@@

#### 2b - Tournament-quality adjustment
Goals in a World Cup tell us more than goals in a friendly. Each match is assigned a tournament weight:

| Competition | Weight |
|---|---|
| FIFA World Cup | 1.00 |
| UEFA Euro / Copa America | 0.90 |
| AFCON / Asian Cup | 0.80 |
| WC Qualifiers | 0.75 |
| Friendlies | 0.40 |

@@LATEX@@
\text{attack\_rate} = \sum_{i} w_i \cdot \frac{\text{goals\_scored}_i}{w_{t,i}}
@@END@@

#### 2c - ELO normalisation
@@LATEX@@
\text{elo\_norm} = \frac{\text{ELO} - \text{ELO}_{\min}}{\text{ELO}_{\max} - \text{ELO}_{\min}}
@@END@@

#### 2d - WC heritage features (jfjelstul/worldcup)
@@LATEX@@
\text{wc\_attack\_rate} = \frac{\sum \text{goals\_for}}{\text{WC matches played}}
@@END@@
@@LATEX@@
\text{wc\_defence\_rate} = \frac{\sum \text{goals\_against}}{\text{WC matches played}}
@@END@@
@@LATEX@@
\text{wc\_stage\_score} = \frac{1}{T}\sum_{t=1}^{T} \text{stage\_depth}(t)
@@END@@

where stage depth is scored: group stage = 0, round of 16 = 1.5, quarter-final = 2, semi-final = 3.5, finalist = 4, winner = 5.

@@NOTE:green@@
**Why this matters:** Brazil WC attack rate is 2.05 goals/game; Morocco is 0.87. These differ substantially from their all-time averages and their current ELO. The GLM can use this to separate general quality from WC pedigree.
@@END@@

#### 2e - Uncertainty score
@@LATEX@@
u = 1 - 0.55 \cdot q - 0.45 \cdot \bar{e}
@@END@@

where q = average tournament quality of recent fixtures and e-bar = average ELO-normalised strength of recent opponents. Both in [0,1], so u in [0.05, 0.95]. A low u (France ~0.11) means high confidence. A high u (Haiti ~0.57) means the model should hedge: that team lambda gets extra random noise in the Monte Carlo step, widening its confidence interval automatically.

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 03@@
### The Poisson distribution -- why goals follow it
Football goals are rare, roughly independent events happening over a fixed 90-minute window -- exactly what Poisson was designed for.

@@LATEX@@
P(X = k \mid \lambda) = \frac{e^{-\lambda}\,\lambda^k}{k!}
\qquad k = 0, 1, 2, 3, \ldots
@@END@@

@@NOTE:blue@@
**Example -- Spain with lambda = 1.8:**
P(0 goals) = e^(-1.8) x 1.8^0 / 1 = 16.5%
P(1 goal)  = e^(-1.8) x 1.8^1 / 1 = 29.8%
P(2 goals) = e^(-1.8) x 1.8^2 / 2 = 26.8%
P(3 goals) = e^(-1.8) x 1.8^3 / 6 = 16.1%
@@END@@

### The GLM -- learning lambda from history
@@LATEX@@
\log(\lambda_{\text{home}}) =
  \beta_0
  + \beta_1 \cdot \text{home\_attack}
  + \beta_2 \cdot \text{away\_defence}
  + \beta_3 \cdot \text{home\_elo\_norm}
  + \beta_4 \cdot \text{away\_elo\_norm}
  + \beta_5 \cdot \text{host\_advantage}
  + \beta_6 \cdot \text{neutral\_venue}
  + \beta_7 \cdot \text{tournament\_weight}
  + \beta_8 \cdot \text{home\_wc\_attack}
  + \beta_9 \cdot \text{away\_wc\_defence}
  + \beta_{10} \cdot \text{home\_wc\_stage}
@@END@@

And symmetrically for the away team (no host_advantage term):

@@LATEX@@
\log(\lambda_{\text{away}}) =
  \gamma_0
  + \gamma_1 \cdot \text{away\_attack}
  + \gamma_2 \cdot \text{home\_defence}
  + \gamma_3 \cdot \text{home\_elo\_norm}
  + \gamma_4 \cdot \text{away\_elo\_norm}
  + \gamma_5 \cdot \text{neutral\_venue}
  + \gamma_6 \cdot \text{tournament\_weight}
  + \gamma_7 \cdot \text{away\_wc\_attack}
  + \gamma_8 \cdot \text{home\_wc\_defence}
  + \gamma_9 \cdot \text{away\_wc\_stage}
@@END@@

The beta and gamma coefficients are learned automatically by fitting the model to all historical match data using maximum likelihood.

@@LATEX@@
\lambda_{\text{home}} = e^{\,\log(\lambda_{\text{home}})}
@@END@@

@@NOTE:green@@
**In plain English:** my model asks, given everything I know about these two teams -- their recent form, world rankings, WC history, and the setting -- what is the average number of goals each team should score? The GLM finds the best-fit answer from a century of match data.
@@END@@

#### Neutral-venue symmetry fix
All WC 2026 group matches are at neutral stadiums. To handle this, I run the prediction twice and average geometrically:

@@LATEX@@
\lambda_A = \sqrt{\lambda_{\text{home}}(A,B) \times \lambda_{\text{away}}(A,B)}
\qquad
\lambda_B = \sqrt{\lambda_{\text{away}}(B,A) \times \lambda_{\text{home}}(B,A)}
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 04@@
### Dixon-Coles correction for low scores
Pure independent Poisson models underestimate 0-0 and 1-1 draws. The Dixon-Coles (1997) correction applies a multiplicative factor tau:

@@LATEX@@
\tau(\,j,\,k\,) =
\begin{cases}
1 - \rho\,\lambda_A\,\lambda_B & j=0,\;k=0 \\
1 + \rho\,\lambda_A            & j=0,\;k=1 \\
1 + \rho\,\lambda_B            & j=1,\;k=0 \\
1 - \rho                        & j=1,\;k=1 \\
1                               & \text{otherwise}
\end{cases}
@@END@@

where rho is a fitted parameter (I use rho = -0.13). The corrected joint probability of scoreline (j, k) becomes:

@@LATEX@@
P(j,k) = \tau(j,k) \times P_{\text{Poisson}}(X{=}j \mid \lambda_A)
                   \times P_{\text{Poisson}}(Y{=}k \mid \lambda_B)
@@END@@

@@NOTE:blue@@
**What this does in practice:** with lambda_A = 1.4 and lambda_B = 1.1, raw Poisson gives P(0-0) = 9.5%. After Dixon-Coles, P(0-0) rises to about 10.9%.
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 05@@
### Tournament form adjustment
I train the GLM on historical data, then correct for current 2026 performance with a multiplicative lambda-adjustment from live ESPN player stats.

#### The weight -- how much to trust the current tournament
@@LATEX@@
w_{\text{form}} = \min\!\left(\frac{n}{3},\; 1\right) \times 0.35
@@END@@

where n is the number of completed tournament matches. After 0 games the weight is 0. After 3+ games it reaches its maximum of 0.35.

#### Attack multiplier
@@LATEX@@
r_{\text{atk}} = \text{clip}\!\left(\frac{\text{goals\_pg}}{\bar{g}},\; 0.5,\; 2.5\right)
@@END@@
@@LATEX@@
m_{\text{atk}} = 1 + w_{\text{form}} \times (r_{\text{atk}} - 1)
@@END@@

A team scoring at exactly the tournament average gets no change. A team scoring twice the average gets a boost of up to +35% on their lambda.

#### Defence multiplier
@@LATEX@@
r_{\text{def}} = \text{clip}\!\left(\frac{\bar{g}}{\text{conceded\_pg}},\; 0.5,\; 2.5\right)
\qquad
m_{\text{def}} = 1 + w_{\text{form}} \times (r_{\text{def}} - 1)
@@END@@

#### Combined effect on lambda
@@LATEX@@
\lambda_A^* = \frac{\lambda_A \times m_{\text{atk},A}}
                   {\max\!\bigl(m_{\text{def},B},\; 0.75\bigr)}
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 06@@
### Injury and suspension adjustment

| Position | Attack penalty | Defence penalty |
|---|---|---|
| Forward | -12% per player | -2% per player |
| Midfielder | -7% per player | -5% per player |
| Defender | -2% per player | -10% per player |
| Goalkeeper | 0% | -6% per player |

@@LATEX@@
m_{\text{atk}} = \prod_{i \in \text{injured}} \left(1 - p_{\text{atk},i}\right)
\qquad
m_{\text{def}} = \prod_{i \in \text{injured}} \left(1 - p_{\text{def},i}\right)
@@END@@

Both are capped at a maximum total penalty of 25% (floor of 0.75), since national teams carry deep squads.

@@LATEX@@
m_{\text{atk}} \geq 0.75 \qquad m_{\text{def}} \geq 0.75
@@END@@

**Opponent ripple effect:** a weakened defence in Team A means Team B faces less resistance:

@@LATEX@@
\lambda_B^* = \frac{\lambda_B}{\max(m_{\text{def},A},\; 0.75)}
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 07@@
### Betting market blend -- DraftKings

#### Step 1 -- Convert American odds to probability
@@LATEX@@
p_{\text{raw}} =
\begin{cases}
\dfrac{100}{|ML| + 100} & \text{if } ML < 0 \text{ (favourite)} \\[8pt]
\dfrac{100}{ML + 100}   & \text{if } ML \geq 0 \text{ (underdog)}
\end{cases}
@@END@@

#### Step 2 -- Remove the vig
@@LATEX@@
p_{\text{fair},i} = \frac{p_{\text{raw},i}}{p_H + p_D + p_A}
\qquad i \in \{H, D, A\}
@@END@@

#### Step 3 -- Recover market-implied lambda
@@LATEX@@
\lambda_{\text{mkt,home}} = \mu_{\text{total}} \times
  \frac{p_{\text{fair},H} + 0.5\,p_{\text{fair},D}}
       {p_{\text{fair},H} + p_{\text{fair},D} + p_{\text{fair},A}}
@@END@@

#### Step 4 -- Geometric blend
@@LATEX@@
\lambda_{\text{blend}} = \lambda_{\text{model}}^{\,(1-w)} \times \lambda_{\text{market}}^{\,w}
@@END@@

where w in [0,1] is the market-influence slider. At w=0: pure statistical model. At w=1: pure market. At w=0.5 (default): equal blend.

@@NOTE:green@@
**Why geometric and not arithmetic?** Lambda is a rate -- doubling it and halving it should be symmetric. The geometric mean respects this symmetry. Example: model says lambda = 1.0, market says lambda = 2.0. Arithmetic mean = 1.5. Geometric mean = sqrt(2) = 1.41.
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 08@@
### Polymarket -- prediction market tournament-winner odds
Polymarket runs binary prediction markets: Will X win the 2026 FIFA World Cup? Each market pays out 1 USDC if Yes, 0 if No. The Yes price equals the implied probability:

@@LATEX@@
P(\text{X wins WC}) \approx \text{price}_{\text{Yes}}
@@END@@

I display these as a relative strength ratio between the two teams:

@@LATEX@@
\text{edge ratio} = \frac{P(\text{Team A wins WC})}{P(\text{Team B wins WC})}
@@END@@

These are tournament-winner probabilities, not match-specific win probabilities, so I show them as supplementary context only and do not blend them into lambda.

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 09@@
### Monte Carlo simulation -- playing the match 10,000 times

@@CODE@@
for i in 1 ... 10,000:

    # Step 1 -- add extra random wobble for uncertain teams
    noise_A ~ Normal(0, u_A x lambda_A*)
    noise_B ~ Normal(0, u_B x lambda_B*)
    lambda_A_noisy = max(lambda_A* + noise_A,  0.05)
    lambda_B_noisy = max(lambda_B* + noise_B,  0.05)

    # Step 2 -- draw a random scoreline from each team Poisson
    goals_A ~ Poisson(lambda_A_noisy)
    goals_B ~ Poisson(lambda_B_noisy)

    # Step 3 -- reweight by Dixon-Coles correction factor
    weight_i = tau(goals_A, goals_B)

    # Step 4 -- record
    results[i] = (goals_A, goals_B, weight_i)
@@END@@

After 10,000 weighted simulations I read off:

@@LATEX@@
P(\text{A wins}) = \frac{\sum_{i:\; g_{A,i} > g_{B,i}} w_i}{\sum_i w_i}
\qquad
\overline{GD} = \frac{\sum_i w_i \cdot (g_{A,i} - g_{B,i})}{\sum_i w_i}
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 10@@
### Skellam distribution -- exact confidence interval on goal difference
If X ~ Poisson(lambda_A) and Y ~ Poisson(lambda_B) are independent, the goal differential D = X - Y follows the Skellam distribution:

@@LATEX@@
P(D = k) = e^{-(\lambda_A + \lambda_B)}
\left(\frac{\lambda_A}{\lambda_B}\right)^{k/2}
I_{|k|}\!\left(2\sqrt{\lambda_A \lambda_B}\right)
@@END@@

The mean and variance are elegantly simple:

@@LATEX@@
E[D] = \lambda_A - \lambda_B
\qquad
\text{Var}[D] = \lambda_A + \lambda_B
@@END@@
@@LATEX@@
\text{95\% CI} \approx
\left[\,(\lambda_A - \lambda_B) \;-\; 1.96\sqrt{\lambda_A + \lambda_B},\;\;
       (\lambda_A - \lambda_B) \;+\; 1.96\sqrt{\lambda_A + \lambda_B}\,\right]
@@END@@

@@NOTE:blue@@
**Example:** if lambda_A = 1.6 and lambda_B = 1.1:
E[D] = 0.5 (Team A expected to win by half a goal)
Var[D] = 2.7, SD = 1.64
95% CI = [0.5 - 1.96 x 1.64, 0.5 + 1.96 x 1.64] = [-2.7, +3.7]
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 11@@
### How to read every number on the main page

| What you see | What it means | How it is computed |
|---|---|---|
| **Team A Win / Draw / Team B Win %** | How often each outcome occurred across 10,000 simulations | Weighted count of simulated results |
| **Expected Goal Diff** | Average winning margin | lambda_A* minus lambda_B* |
| **+/- Margin of Error** | Half-width of the 95% CI | (CI_high minus CI_low) / 2 from Skellam |
| **lambda (xG) values** | Final expected goals after all adjustments | GLM output x form multipliers x injury multipliers |
| **Most likely scoreline** | The exact score with highest simulated frequency | Mode of weighted (goals_A, goals_B) pairs |
| **Outcome donut** | Win/draw/loss split as a ring chart | Same three percentages, visualised |
| **Goal Differential chart** | Full distribution of goal margins, 95% CI shaded | Histogram of 10,000 GD values |
| **Goals Scored chart** | How likely each team is to score 0, 1, 2, 3... | Poisson PMF using final lambda values |
| **Scoreline heatmap** | Probability of every exact score | Joint Poisson P(j) x P(k) x tau(j,k) |
| **Model vs Market table** | Pure model / DraftKings / blended, side by side | Three separate lambda-pairs run through the same Skellam CI |
| **Polymarket panel** | Tournament-winner probability per team | Yes prices from Polymarket API, refreshed every 5 min |
| **WC Heritage panel** | Historical WC goals/game, win rate, stage depth | Aggregated from jfjelstul/worldcup 1930-2022 |
| **Squad panel** | Player list with WC 2026 goals/assists/cards | Per-player stats from ESPN match summaries |

@@NOTE:green@@
**Rule of thumb:** look at the win % first to see who is favoured, then check the margin of error. A 60% favourite with +/-0.4 is a confident call; a 60% favourite with +/-1.8 is barely distinguishable from a coin-flip.
@@END@@

;; ---------------------------------------------------------------------------
@@CARD@@
@@BADGE: Step 12@@
### What my model cannot do

- **No starting XI.** ESPN gives me the declared squad (26 players), not the confirmed eleven, so rotation and tactical surprises are invisible to me.
- **No club form.** I train only on international matches.
- **No penalty shootouts.** All predictions cover 90 minutes. The lottery of penalties is not modelled.
- **Goals assumed mostly independent.** Dixon-Coles partially corrects for correlation at 0-0 and 1-1, but extreme scorelines can still have slightly mis-stated draw chances.
- **Thin data = honest but wide uncertainty.** Teams from regions with fewer recorded elite matches carry a high uncertainty score, flagged with a wider margin of error.
- **Polymarket illiquidity.** For teams with small markets, the Yes price may not reflect sharp money and can be noisy, which is why I keep it out of the lambda blend.

@@NOTE:amber@@
**This is a statistical estimate, not a guarantee.** Upsets -- a 20% team beating an 80% favourite -- happen in roughly 1 in 5 such predictions. That is not a model failure; it is what 20% means.
@@END@@

@@FOOTER: Data: ESPN / eloratings.net / jfjelstul/worldcup (GitHub) / Kaggle martj42 / Polymarket - No API key required - Model: Poisson GLM (statsmodels) + Dixon-Coles + Monte Carlo
