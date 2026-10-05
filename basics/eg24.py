import requests

#googledata=requests.get("http://www.google.com")
#print(googledata.text)

todos=requests.get("https://jsonplaceholder.typicode.com/todos/")
print(todos.json())