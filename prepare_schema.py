import sqlite_utils
import os

def get_schema_text(db_path):
    db = sqlite_utils.Database(db_path)
    schema_text = ""
    for table in db.tables:
        schema_text += f"Table: {table.name}\n"
        schema_text += table.schema + "\n\n"
    return schema_text

# Option 1: Relative path (if chinook.db is in the same directory)
# db_path = "chinook.db"

# Option 2: Absolute path (more reliable)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
db_path = os.path.join(script_dir, "chinook.db") # Full path

schema_string = get_schema_text(db_path)
print(schema_string)