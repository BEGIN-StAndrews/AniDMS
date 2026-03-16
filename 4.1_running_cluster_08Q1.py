from datetime import datetime, timedelta
import gc
import os
os.chdir(r'/sharedscratch/ml340/DMSEstimation')

from predictor_projected import Efficient4kmDMSPredictor

model_path = r"/sharedscratch/ml340/DMSEstimation/models/best_stacking_model_2_elasticnet.joblib"
scaler_path = r"/sharedscratch/ml340/DMSEstimation/models/standard_scaler.joblib"
pred = Efficient4kmDMSPredictor(model_path, scaler_path)

start_date = datetime(2008, 1, 1).date()
end_date = datetime(2008, 3, 31).date()

current_date = start_date

while current_date <= end_date:
    print(f"{'='*50}")
    print(f"Processing date: {current_date}")
    print(f"{'='*50}")

    try:
        success = pred.run_prediction_for_date(current_date)
        if success:
            print(f"Successfully processed {current_date}")
        else:
            print(f"Failed to process {current_date}")
    
    except Exception as e:
        print(f"Error processing {current_date}: {e}")
        import traceback
        traceback.print_exc()
    
    current_date += timedelta(days=1)
    gc.collect()
    
pred.save_metadata(start_date, end_date)

gc.collect()
print("All dates processed!")