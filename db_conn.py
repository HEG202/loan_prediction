import pymysql


def open_db():
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="230223",
        database="loan_db",
        charset="utf8mb4"
    )
    cur = conn.cursor(pymysql.cursors.DictCursor)
    return conn, cur


def close_db(conn, cur):
    cur.close()
    conn.close()