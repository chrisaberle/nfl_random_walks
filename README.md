# NFL Random Walks Script

A simple heuristic I developed to guide my selections in and NFL Survivor League.

## Approach

We begin with a Pandas dataframe where each row represents a team and the columns represent their current
betting market spreads. The script then picks a random walk through the season, scores it according to
a heuristic, and keeps track of the best scoring heuristic across all iterations.

Because the potential number of walks is so large (over a septillion in week one) I've done some work to reduce the
solution space:

1. Weeks are ordered according to their difficulty. Weeks where there are few good choices are processed first.
2. Weeks are also ordered according to average washout rates across many survivor pools.
3. In any given week only the best three spreads are considered for selection. In testing (vs. top 4 or 5) this yielded the best results.

The scoring heuristic is some balance of average spread (maximizing chances of survival across the whole season),
while avoiding both very high spreads and very low spreads. These weights are ALPHA, BETA, and GAMMA, which are
defined in an `.env` file but can easily be hardcoded.

I currently run 200,000 walks. Usually the solution converges well before that.

## Usage

The script itself is easy enough to run. I use Beautiful Soup to extract current spreads from my favorite website. I define weights and define the teams I've already chosen for the season, then let 'er rip. The console prints basic information about the progress of the exploration:

```bash
The week order is ['1', '2', '5', '3', '4', '7', '6', '9', '8', '13', '12', '11', '10', '17', '14', '15', '16', '18']
Iteration: 100 | Best score: 2.3555555555555556 | Best spread: -89.0 | Largest spread: -8.0 | Smallest spread: -2.5 | Average spread: -4.944444444444445 | Best walk: ['WAS', 'DEN', 'CIN', 'SEA', 'NO', 'BUF', 'ATL', 'CLE', 'LAC', 'PIT', 'NYJ', 'JAX', 'MIN', 'SF', 'NYG', 'GB', 'CHI', 'CAR'] |
```
After it finishes it prints a tabular representation of the picks in each week. I used polars for this because pandas tables are ugly. The table is designed to make it easy for me to audit results, it's not exactly useful otherwise:

```bash
Final best sum: -89.0
Final best walk: ['WAS', 'DEN', 'CIN', 'SEA', 'NO', 'BUF', 'ATL', 'CLE', 'LAC', 'PIT', 'NYJ', 'JAX', 'MIN', 'SF', 'NYG', 'GB', 'CHI', 'CAR']
shape: (27, 19)
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ TEA ┆ 1   ┆ 2   ┆ 3   ┆ 4   ┆ 5   ┆ 6   ┆ 7   ┆ 8   ┆ 9   ┆ 10  ┆ 11  ┆ 12  ┆ 13  ┆ 14  ┆ 15  ┆ 16  ┆ 17  ┆ 18  │
│ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- │
│ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str ┆ str │
╞═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╡
│ ARI ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ ARI ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ ATL ┆     ┆     ┆     ┆     ┆     ┆ ATL ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ (-3 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ .0) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ BUF ┆     ┆     ┆     ┆     ┆     ┆     ┆ BUF ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-3 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .5) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ CAR ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ CAR │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-3 │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .0) │
│ CHI ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ CHI ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-4 ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .5) ┆     ┆     │
│ CIN ┆     ┆     ┆     ┆     ┆ CIN ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆ (-7 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆ .5) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ CLE ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ CLE ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-8 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .0) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ DEN ┆     ┆ DEN ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆ (-4 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆ .0) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ DET ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ GB  ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ GB  ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-4 ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .0) ┆     ┆     ┆     │
│ HOU ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ IND ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ JAX ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ JAX ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-6 ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .0) ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ LAC ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ LAC ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-6 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .5) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ LAR ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ MIN ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ MIN ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-2 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .5) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ NE  ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ NO  ┆     ┆     ┆     ┆ NO  ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆ (-6 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆ .5) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ NYG ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ NYG ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-2 ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .5) ┆     ┆     ┆     ┆     │
│ NYJ ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ NYJ ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-3 ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .0) ┆     ┆     ┆     ┆     ┆     ┆     │
│ PIT ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ PIT ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-7 ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .0) ┆     ┆     ┆     ┆     ┆     │
│ SEA ┆     ┆     ┆ SEA ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆ (-6 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆     ┆     ┆ .0) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ SF  ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ SF  ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ (-4 ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆ .5) ┆     │
│ TB  ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ TEN ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│ WAS ┆ WAS ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆ (-7 ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
│     ┆ .0) ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     ┆     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

I've masked the website and weights in the public repo so as not to take the fun out of the pool I participate in case participants want to peek.

## Improvements

Random walks are a pretty blunt instrument and the results are highly dependent on the strength of my heuristic, which
is not very strong. Some known weaknesses I didn't have time to address before the season started:

1. **Future Value** is easy enough to calculate and often used in pool strategies but I haven't used it here.
2. Related to #1, even with the aggressive washout rate week ordering this approach still uses good teams early. Because my pool allows two losses in weeks 1-11 it's especially suboptimal.
3. Only a single feature, spread, is used for decision making. It is assumed that spread accounts for many other factors (home/away, streak, recent injuries, &c.) but this limits strategies that can be discovered.
4. When one looks at spread vs. actual win statistics historically, spread is a pretty poor measure of actual probability. Converting the script would yield better real-world results.
