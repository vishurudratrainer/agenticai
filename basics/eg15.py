#Dictionary is used for mapping
#key value pair
#Key cannot be duplicated
#insert/update/delete

country={}
country=dict()
print(country)
country["US"]="United states"#insertion
country["IN"]="India"
country["IN"]="Bharat"#Updating
print(country)
for k,v in country.items():
    print(k,v)
del country["US"]
print(country)