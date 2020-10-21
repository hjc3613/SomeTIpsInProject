# Python 3 program to find the stem 
# of given list of words 

# function to find the stem (longest 
# common substring) from the string array 
def findstem(arr): 

	# Determine size of the array 
	n = len(arr) 

	# Take first word from array 
	# as reference 
	s = arr[0] 
	l = len(s) 

	res = "" 

	for i in range( l) : 
		for j in range( i + 1, l + 1) : 

			# generating all possible substrings 
			# of our reference string arr[0] i.e s 
			stem = s[i:j] 
			k = 1
			for k in range(1, n): 

				# Check if the generated stem is 
				# common to all words 
				if stem not in arr[k]: 
					break
			
			# If current substring is present in 
			# all strings and its length is greater 
			# than current result 
			if (k + 1 == n and len(res) < len(stem)): 
				res = stem 

	return res 

# Driver Code 
if __name__ == "__main__": 
	
	arr = ["武汉普仁医院", "——— 武汉普仁医院****",
           "武汉普仁医院同济分院", "武汉普仁医院检验单"]
	stems = findstem(arr) 
	print(stems) 

# This code is contributed by ita_c 
