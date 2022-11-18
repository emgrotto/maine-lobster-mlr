# maine-lobster-mlr
Analysis of temperature data effects on lobster landings

To run the project using Docker:

```
docker build -t lobster .
docker run lobster
docker run -it -v $(pwd)/etl/:/etl/ lobster bash
```
