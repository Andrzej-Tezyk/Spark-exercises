'''Spark exercises found on github of professor of Université Lumière Lyon: https://github.com/andfanilo/pyspark-streamlit-tutorial/tree/master'''

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark import SparkContext
from pyspark.rdd import RDD


def rdd_from_list(sc: SparkContext, n: int) -> RDD:
    """Return a RDD consisting of elements from 1 to n.
    For now we assume we will always get n > 1, no need to test for the exception nor raise an Exception.
    """

    return sc.parallelize(range(0, n))


def load_file_to_rdd(sc: SparkContext, path: str) -> RDD:
    """Create a RDD by loading an external file. We don't expect any formatting nor processing here.
    You don't need to raise an exception if the file does not exist.
    """
    
    return sc.textFile('data/titanic.csv')


def op1(sc: SparkContext, mat: RDD) -> RDD:
    """Multiply the first coordinate by 2, remove 3 to the second"""

    return mat.map(lambda x: (x[0] * 2, x[1] - 3))


def op2(sc: SparkContext, sentences: RDD) -> RDD:
    """Return all words contained in the sentences."""

    return sentences.collect()


def op3(sc: SparkContext, numbers: RDD) -> RDD:
    """Return all numbers contained in the RDD that are odd."""
    
    return numbers.filter(lambda x: x % 2 != 0)


def op4(sc: SparkContext, numbers: RDD) -> RDD:
    """Return the sum of all squared odd numbers"""
    
    return numbers.reduce(lambda x, y: x + y)


def wordcount(sc: SparkContext, sentences: RDD) -> RDD:
    """Given a RDD of sentences, return the wordcount, after splitting sentences per whitespace."""
    
    return sentences.flatMap(lambda x: x.split()).count()


def mean_grade_per_gender(sc: SparkContext, genders: RDD, grades: RDD) -> RDD:
    """Given a RDD of studentID to grades and studentID to gender, compute mean grade for each gender returned as paired RDD.
    Assume all studentIDs are present in both RDDs, making inner join possible, no need to check that.
    """

    joined = genders.join(grades)

    gender_grades = joined.map(lambda x: (x[1][0], x[1][1]))

    totals_and_counts = gender_grades.aggregateByKey(
        (0, 0),
        lambda acc, grade: (acc[0] + grade, acc[1] + 1),
        lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
    )

    mean_grades = totals_and_counts.mapValues(lambda x: x[0] / x[1])

    return mean_grades


def filter_header(sc: SparkContext, rdd: RDD) -> RDD:
    """From a RDD of lines from a text file, remove the first line."""

    header = rdd.first()
    
    return rdd.filter(lambda line: line != header)


def county_count(sc: SparkContext, rdd: RDD) -> RDD:
    """Return a RDD of key,value with county as key, count as values"""
    
    county_pairs = rdd.map(lambda record: (record['county'], 1))

    return county_pairs.reduceByKey(lambda x, y: x + y)


def load_and_process_data(sc: SparkContext):
    """Stub for loading and processing data."""
    spark = SparkSession(sc)

    data = spark.read.csv('path/to/FL_insurance.csv', header=True, inferSchema=True)
    processed_data = data.groupBy("county").agg(count("*").alias("occurrences"))
    return processed_data

def bar_chart_county(sc: SparkContext) -> None:
    """Display a bar chart for the number of occurences for each county
    with Matplotlib, Bokeh, Plotly or Altair...

    Don't return anything, just display in the function directly.

    Load and process the data by using the methods you defined previously.
    """
    
    processed_data = load_and_process_data(sc)
    
    # data for plotting
    county_data = processed_data.collect()
    counties = [row["county"] for row in county_data]
    occurrences = [row["occurrences"] for row in county_data]
    
    plt.figure(figsize=(10, 6))
    plt.bar(counties, occurrences, color='skyblue')
    plt.xlabel('County')
    plt.ylabel('Number of Occurrences')
    plt.title('Number of Occurrences per County')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()