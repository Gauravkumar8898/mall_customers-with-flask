from src.pipeline.mall_customers_pipeline import pipline
from src.flask.app import app
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Enter 1 for model training  or 2 for prediction:-")
    val = abs(int(input()))
    if val == 1:
        pipline()
    elif val == 2:
        app.run(debug=True, host='0.0.0.0', port=port)
