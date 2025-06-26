from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import sqlite3
import os

app = FastAPI()
DB_PATH = "postprocessing/review_queue.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS review_queue
                 (wafer_id TEXT, img TEXT, label TEXT, reviewed INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def review_queue():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT rowid, wafer_id, img FROM review_queue WHERE reviewed=0")
    rows = c.fetchall()
    conn.close()
    html = "<h1>Manual Review Queue</h1>"
    for row in rows:
        html += f"<div><img src='data:image/png;base64,{row[2]}' width=128 /><br>"
        html += f"Wafer ID: {row[1]}<form method='post' action='/review'><input type='hidden' name='rowid' value='{row[0]}' />"
        html += "<button name='label' value='defect'>Defect</button><button name='label' value='no_defect'>No Defect</button></form></div><hr>"
    return html

@app.post("/review")
def review(rowid: int = Form(...), label: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE review_queue SET label=?, reviewed=1 WHERE rowid=?", (label, rowid))
    conn.commit()
    conn.close()
    return HTMLResponse("<p>Label updated. <a href='/'>Back</a></p>")
