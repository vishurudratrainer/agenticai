mark=int(input("Enter mark"))
grade=""
if mark < 0 or mark > 100:
    grade="Invalid"
elif mark >=80:
    grade="A"
elif mark >=60:
    grade="B"
elif mark >= 40:
    grade="C"
else:
    grade="D"

print("grade is",grade,"for mark",mark)