import data_collection
import data_cleaning
import generate_synthetic
import database_setup
import model_training
import estimator
import value_metrics
import spec_classifier
import spec_regression
import data_visualization


# runs every pipeline stage in order
# data_collection also ingests the daily real prices from hardwaredealsco
# generate_synthetic builds the synthetic series and bridges in the real_blended rows

def main():
    print("running data collection")
    data_collection.main()

    print("running data cleaning")
    data_cleaning.main()

    print("running generate synthetic")
    generate_synthetic.main()

    print("running database setup")
    database_setup.main()

    print("running model training")
    model_training.main()

    print("running estimator")
    estimator.main()

    print("running value metrics")
    value_metrics.main()

    print("running spec classifier")
    spec_classifier.main()

    print("running spec regression (cross)")
    spec_regression.main()

    print("running data visualization")
    data_visualization.main()


if __name__ == "__main__":
    main()
