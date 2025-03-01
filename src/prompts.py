markdown_to_csv_prompt = '''
### **Instruction**  
Your task is to extract the table from the input text and convert it into CSV format with strict adherence to formatting rules.  

#### **Requirements:**  
- Extract all contents from the column **"Business Requirement"**.  
- Ensure no data is omitted or altered.  
- Use **"^^"** as the delimiter between values.  
- The final output must include the header: **Business Requirements**.  
- Double-check that the delimiter is correctly applied throughout the table.  
- If no table is found, output exactly: **"NO_TABLE"** (without quotes).  
- **Output only the CSV data**, with no additional text, explanations, or formatting artifacts.  

#### **Validation Checks:**  
1. Verify that the table is fully extracted, including all rows under **"Business Requirement"**.  
2. Ensure the delimiter **"^^"** is consistently used between values.  
3. Do not include any extra text, metadata, or formatting outside of the CSV structure.  
4. Remove any text outside of the table.

**Final Reminder:** Double-check the delimiter usage and table extraction accuracy before outputting the result.

'''