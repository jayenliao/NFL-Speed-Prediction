# NFL-Speed-Prediction

This is the final project of "Time Series Analysis", a course at the programme of Data and Business Analytics at Rennes School of Business, France. This study aims to use two types of method to predict NFL players' speed during each play in games, including ARIMA and LSTM.

## Authors
- Jay Chiehen Liao (jay-chiehen.liao@rennes-sb.com)
- Chin-Ching Yang (chin-ching.yang@rennes-sb.com)

## Data

Please go [here](https://www.kaggle.com/c/nfl-big-data-bowl-2022/data) to access the datasets. We only use the tracking datasets.

Here is the description of the whole original dataset:

- __Game data__: The `games.csv` contains the teams playing in each game. The key variable is `gameId`.
- __Play data__: The `plays.csv` file contains play-level information from each game. The key variables are `gameId` and `playId`.
- __Player data__: The `players.csv` file contains player-level information from players that participated in any of the tracking data files. The key variable is `nflId`.
- __Tracking data__: Files `tracking[season].csv` contain player tracking data from season. The key variables are `gameId`, `playId`, and `nflId`.
- __PFF Scouting data__: The `PFFScoutingData.csv` file contains play-level scouting information for each game. The key variables are `gameId` and `playId`.

The original dataset is extremely abundant, which is more than 5GB in total. There are 12,777,351, 12,170,933, and 11,821,701 observations in 2018, 2019, and 2020, respectively. One observation is a record of the player's location (x and y), speed, direction, orientation, and direction, ... etc, with the timestamp. Each observation was recorded by 0.1 second. That is, a time series with 100 rows (length=100) lasted 100*0.1=10 seconds.

In each data frame of these three years, there are several games which consist of several plays. And there are records of several players in each play. There are 253, 255, and 256 games in 2018, 2019, and 2020, respectively. To make the analysis more efficient, we decided to pick one game for each year, which still consists of lots of time series. Since the variation of the count of plays is quite large, it is not appropriate to randomly pick one game for each year. We found that the distributions of plays counts in both 2018 and 2019 are symmetric and that in 2020 are a little bit left-skewed. To make the selected games representative, we picked the game with the count of plays corresponding to the mode of the distribution (either the mean of the median may be biased). If there was not only one mode, a mode was randomly selected. Finally, game 2018100707 with 45,678 observations, game 2019090804 with 37,858 observations, game 2020102503 with 50,462 observations were selected. The reduced dataset is about 19.9MB in total.

## Files

- `./output/` contains png and txt files of results of LSTMs
- `./main.py`: the main procedure of LSTM
- `./args.py`: the definiton of arguments for training LSTM
- `./model.py`: the stucture of LSTM and the training procedure
- `./utils.py`: functions for data loading, preprocessing, and visualization for LSTMs
- `./LSTM-results.ipynb`: the summaried visualization of LSTMs.
- `./EDA-and-ARIMA.ipynb`: exploratory data analysis and all procefures of ARIMA
- `./data_preprocessing.ipynb`: generating smaller dataset

## Usage of training LSTM

- Training with CPU

```
python3 main.py --device 'cpu'
```

- Training with GPU

```
python3 main.py --device 'cuda'
```

- Change #epochs

```
python3 main.py --epochs 1000
```

For more details, please check `./args.py`.
