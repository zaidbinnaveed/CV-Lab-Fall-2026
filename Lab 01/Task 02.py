students = {
    "S001": {"Name": "Ali", "Major": "CS", "Grades": [85, 90, 78]},
    "S002": {"Name": "Sara", "Major": "EE", "Grades": [92, 88, 95]},
    "S003": {"Name": "Ahmed", "Major": "CS", "Grades": [70, 65, 80]},
}


def top_student(records):
    best_id = None
    best_avg = -1
    for sid, info in records.items():
        avg = sum(info["Grades"]) / len(info["Grades"])
        if avg > best_avg:
            best_avg = avg
            best_id = sid
    return records[best_id]["Name"]


def students_by_major(records, major):
    for sid, info in records.items():
        if info["Major"] == major:
            print(info["Name"])


print("Top student:", top_student(students))
print("CS students:")
students_by_major(students, "CS")
