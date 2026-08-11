import sqlite3, sqlite_vec

db = sqlite3.connect("ropa.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

version, = db.execute("select vec_version()").fetchone()
print("sqlite-vec OK:", version)