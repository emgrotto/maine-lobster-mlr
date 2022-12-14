# maine-lobster-mlr
Analysis of temperature data effects on lobster landings


## Running the project

To run the project locally, make sure you have the packages in `requirements.txt` installed and then run:
```
sh run.sh
```

To run the project using Docker:

```
docker build -t lobster .
docker run lobster
```

I have added the results of the most recent execution of the project in the [results.txt](./results.txt)


## Structure of the project
```
.
├── data/                      # Data used by project 
├── paper_resources/           # Images generated by project that are potentially useful for the paper
├── project/                   # project code
├── viewing/                   # Data in a more viewable format
├── Data_exploration.ipynb     
├── Dockerfile
├── README.md                   
├── requirements.txt
├── results.txt
└── run.sh
```
