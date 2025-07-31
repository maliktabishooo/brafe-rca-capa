import os
import streamlit.web.cli as stcli
import sys

def handler(event, context):
    sys.argv = ["streamlit", "run", "app.py", "--server.port", str(os.getenv("PORT", 8501))]
    stcli.main()
    return {"statusCode": 200}
