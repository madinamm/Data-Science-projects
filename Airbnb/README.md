## Project Motivation
In this Analysis we would like to develop a mechanism suggesting relevant pricing of Airbnb listing (property) in Amsterdam based on the major characteristics of a property:
> - We will start from identifying characteristics of a listing (e.g. rating, number of bedrooms etc.), which have the highest correlation to price;
> 
> - Predict pricing based on the selected characteristics of a train dataset;
> 
> - Compare results of the predicted prices (suggested pricing of a listing) against test data using R-squared (R²) statistical measure.

## How to interact with the project
The Analysis is based on Airbnb data for Amsterdam in 2024.

You can access raw data either via:
> - Airbnb website directly for the compressed csv file using https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2024-09-05/data/listings.csv.gz
>
> - download csv file, which is part of the same folder of the current repository

We used pandas, matplotlib, sklearn, seaborn and numpy libraries.

## Key insights and conclusion
Our Linear Regression model turned to be reasonably well in predicting prices, especially in a lower to medium ranges. This could indicate that if this machanism would have been implemented and be in place at Airbnb, it could potentially prevent hosts from indicating prices with extreme or way above market average prices (potentially even prevent from manual mistakes in pricing and/or suggesting prices for the missing values as a recommendation).

Additional insights:
- Number of bedrooms in a listing and maximum capacity of a listing (number of accommodates) have the highest positive correlation to prices;
- Private room type has the highest negative correlation to prices (as a context other room types are: Hotel room, Shared room types, Entire home/apt);
- Number of reviews have negative small correlation to price, while review scores themselves have a positive correlation to prices, indicating that the higher the review the higher price could be generally for the selected listings. Potentially properties with high number of reviews are listed for more frequent rent and thus are not being considered as an expensive property.

This Analysis was made possible thanks to the courtesy of Airbnb in sharing Amsterdam listings data and a great content of Udacity Nanodegree program.
