# Power BI Dashboard Blueprint

Import `school_attendance_performance.csv` into Power BI.

## KPI Cards
- Total Students = COUNT(Student_ID)
- Average Attendance = AVERAGE(Attendance_Percentage)
- Average Marks = AVERAGE(Average_Marks)
- Average Assignment Score = AVERAGE(Assignment_Score)

## Recommended Visuals
1. Scatter chart:
   - X-axis: Attendance_Percentage
   - Y-axis: Average_Marks
   - Legend: Attendance_Category
2. Clustered column chart:
   - Axis: Attendance_Category
   - Values: Average_Marks
3. Column chart:
   - Axis: Class
   - Values: Attendance_Percentage
4. Column chart:
   - Axis: subject
   - Values: average marks
5. Table:
   - Student_ID
   - Class
   - Attendance_Percentage
   - Average_Marks
   - Attendance_Category

## Slicers
- Class
- Gender
- Attendance_Category

## Suggested Dashboard Title
Attendance & Performance Pattern Analysis in Schools
