import feature_extraction
import light_gbm
import catboost_model
import stack

def main():
    feature_extraction.run_all()
    light_gbm.lgbm()
    catboost_model.catb()
    stack.weighted_stack()


if __name__ == '__main__':
    main()