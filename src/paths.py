"""
This script is used to dynamically allocate paths to the Database.
"""

import os
import platform


CSV_PATH = ""
RAW_PATH = ""


def find_paths():
    print("")
    print("Allocating paths", end = '\r')

    # directory where file recides
    pwd = os.path.dirname(os.path.abspath(__file__))
    
    # directory where we want to search for Database
    wd = os.path.dirname(os.path.dirname(pwd))
    target_name = "Database"
    target_dir = ""

    global RAW_PATH
    global CSV_PATH

    for root, dirs, _ in os.walk(wd):
        for dir_name in dirs:
            if dir_name.lower() == target_name.lower():
                target_dir = os.path.join(root, dir_name)

    if target_dir == "":
        raise ValueError("Database not found")

    for root, dirs, _ in os.walk(target_dir):
        for dir_name in dirs:
            if dir_name.lower() == "raw_data":
                RAW_PATH = os.path.join(root, dir_name)

    for root, dirs, _ in os.walk(target_dir):
        for dir_name in dirs:
            if dir_name.lower() == "csv_data":
                CSV_PATH = os.path.join(root, dir_name)

    if RAW_PATH == "":
        raise ValueError("RAW Database not found.")

    if CSV_PATH == "":
        raise ValueError("CSV Database not found.")

    if platform.system() == 'Windows':
        RAW_PATH += "\\"
        CSV_PATH += "\\"
    else:
        RAW_PATH += "/"
        CSV_PATH += "/"
        
