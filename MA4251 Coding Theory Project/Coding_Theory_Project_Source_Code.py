import tkinter as tk

window = tk.Tk()
window.geometry("800x600")
window.title("MA4251 Coding Theory Project")
window.resizable(False,False)
font1 = ['Open Sans',14]

#####################################################
# Main Menu

main_menu = tk.Frame(window)
window.grid_columnconfigure(0, weight=1)

basis_calc_page = tk.Frame(window)
enc_dec_page = tk.Frame(window)

basis_calc_page.grid(row=0, column=0, sticky='nsew')  
enc_dec_page.grid(row=0, column=0, sticky='nsew') 
main_menu.grid(row=0, column=0, sticky='nsew')
basis_calc_page.grid_columnconfigure(0, weight=1)  
enc_dec_page.grid_columnconfigure(0, weight=1)  
main_menu.grid_columnconfigure(0, weight=1)
main_menu.tkraise()

# Title
lb1 = tk.Label(main_menu, text="MA4251 Coding Theory Project ", font=['Open Sans',24])
lb1.grid(row=0, column=0, pady=(180,0))
lb2 = tk.Label(main_menu, text="Juan Farrell P 10119047", font=font1)
lb2.grid(row=1, column=0)
lb3 = tk.Label(main_menu, text="Okta Pratama 10119056", font=font1)
lb3.grid(row=2, column=0)

button_frame = tk.LabelFrame(main_menu, borderwidth=0,  highlightthickness=0)
button_frame.grid(row=3, column=0)

basis_calc_btn = tk.Button(button_frame, text="Basis Calculator", font=font1, command=lambda:basis_calc())
basis_calc_btn.grid(row=1, column=0, padx=10, pady=10)
basis_calc_btn = tk.Button(button_frame, text="Encoding & Decoding", font=font1, command=lambda:enc_dec())
basis_calc_btn.grid(row=1, column=1, padx=10, pady=10)

def basis_calc():
	basis_calc_page.tkraise()
	
def enc_dec():
	enc_dec_page.tkraise()

def back():
	main_menu.tkraise()

#####################################################
# BASIS CALCULATOR PAGE

input_frame1 = tk.LabelFrame(basis_calc_page, text="Input")
input_frame1.grid(row=0, column=0, pady=(50,10))

button_frame1 = tk.LabelFrame(input_frame1, borderwidth=0,  highlightthickness=0)
button_frame1.grid(row=2, columnspan=3)

q_label = tk.Label(input_frame1, text="Alphabet size", font=font1)
S_label = tk.Label(input_frame1, text="Code", font=font1)

q_input = tk.StringVar(value="5")
S_input = tk.StringVar(value="312110441,423130030,034012301,124224011,243441430,132032032,402030323,041231032")

q_entry = tk.Entry(input_frame1, font=font1, textvariable=q_input)
S_entry = tk.Entry(input_frame1, font=font1, textvariable=S_input, width='50')

btn1 = tk.Button(button_frame1,text="Algorithm 4.1",font=font1,command=lambda:algorithm_41())
btn2 = tk.Button(button_frame1,text="Algorithm 4.2",font=font1,command=lambda:algorithm_42())
btn3 = tk.Button(button_frame1,text="Algorithm 4.3",font=font1,command=lambda:algorithm_43())

q_label.grid(row=0, column=0, sticky='E', padx=10)
S_label.grid(row=1, column=0, sticky='E', padx=10)

q_entry.grid(row=0, column=1, sticky='W')
S_entry.grid(row=1, column=1, sticky='W', columnspan=2, padx=(0,10))

btn1.grid(row=0, column=0, padx=10, pady=10)
btn2.grid(row=0, column=1, padx=10, pady=10)
btn3.grid(row=0, column=2, padx=10, pady=10)

# Displaying Output
output_frame1 = tk.LabelFrame(basis_calc_page, text="Output")
output_frame1.grid(row=1, column=0)

output1 = tk.Label(output_frame1, height=3, font=font1, justify='center',wraplength=550)
output1.grid(row=3, column=0, padx=10, columnspan=3)

bck = tk.Button(basis_calc_page,text="Back",font=font1,command=lambda:back())
bck.grid(row=3, column=0, padx=10, pady=10)

#####################################################
# ENCODING & DECODING PAGE

# ENCODING
input_frame2 = tk.LabelFrame(enc_dec_page, text="Input")
input_frame2.grid(row=0, column=0, pady=(30,10))

q2_label = tk.Label(input_frame2, text="Alphabet size", font=font1)
S2_label = tk.Label(input_frame2, text="Code", font=font1)
u_label = tk.Label(input_frame2, text="Word", font=font1)

q2_input = tk.StringVar(value="3")
S2_input = tk.StringVar(value="10210,01021,00201")
u_input = tk.StringVar(value="101")

# OTHER TESTING EXAMPLE
#q2_input = tk.StringVar(value="5")
#S2_input = tk.StringVar(value="312110441,423130030,034012301,124224011,243441430,132032032,402030323,041231032")
#u_input = tk.StringVar(value="10123401")

q2_entry = tk.Entry(input_frame2, font=font1, textvariable=q2_input)
S2_entry = tk.Entry(input_frame2, font=font1, textvariable=S2_input, width='50')
u_entry = tk.Entry(input_frame2, font=font1, textvariable=u_input)

btn4 = tk.Button(input_frame2,text="Encode",font=font1,command=lambda:encode())

q2_label.grid(row=0, column=0, sticky='E', padx=10)
S2_label.grid(row=1, column=0, sticky='E', padx=10)
u_label.grid(row=2, column=0, sticky='E', padx=10)

q2_entry.grid(row=0, column=1, sticky='W')
S2_entry.grid(row=1, column=1, sticky='W', columnspan=2, padx=(0,10))
u_entry.grid(row=2, column=1, sticky='W')

btn4.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

# Displaying Output
output_frame2 = tk.LabelFrame(enc_dec_page, text="Output")
output_frame2.grid(row=1, column=0)

output2 = tk.Label(output_frame2, height=2, font=font1, justify='center',width=40)
output2.grid(row=3, column=0, padx=10, columnspan=3, sticky='w')


# DECODING
input_frame3 = tk.LabelFrame(enc_dec_page, text="Input")
input_frame3.grid(row=2, column=0, pady=(10,10))

v_label = tk.Label(input_frame3, text="Sent codeword", font=font1)
v_input = tk.StringVar(value="10121")
v_entry = tk.Entry(input_frame3, font=font1, textvariable=v_input)
v_label.grid(row=0, column=0, sticky='E', padx=10)
v_entry.grid(row=0, column=1, sticky='W')
btn5 = tk.Button(input_frame3,text="Decode",font=font1,command=lambda:decode())
btn5.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Displaying Output
output_frame3 = tk.LabelFrame(enc_dec_page, text="Output")
output_frame3.grid(row=3, column=0)

output3 = tk.Label(output_frame3, height=2, font=font1, justify='center',width=40)
output3.grid(row=3, column=0, padx=10, columnspan=3, sticky='w')

bck = tk.Button(enc_dec_page,text="Back",font=font1,command=lambda:back())
bck.grid(row=4, column=0, padx=10, pady=10)

#####################################################

def str_set_to_matrix(S):
	''' 
	Forming a matrix A whose rows are the codewords in S
	S: an array of codewords
	'''
	n = len(S[0])
	K = len(S)
	A = [[0 for k in range(0,n)] for i in range(0,K)]
	for i in range(0,K):
		for k in range (0,n):
			A[i][k] = int(S[i][k])
	return A
 
#####################################################
# ALGORITHM 4.1

def algorithm_41():
	'''
	Performing Algorithm_4.1 to obtain a basis
	q: the number of alphabets in code
	S: an array of codewords
	'''
	q = int(q_input.get())
	S = list(map(str,S_input.get().split(',')))
	
	K = len(S)
	n = len(S[0])

	A = str_set_to_matrix(S)

	## Finding REF (The idea is similar to Gauss-Jordan elimination)
	p = min(n,K)
	for k in range(K):
		if k < p:
			for j in range(k,K):
				# Swapping two rows to obtain 1 as leading entry
				if A[j][k]!=1:
					m = k
					for i in range(k+1,K):
						if A[i][k] > A[m][k]:
							m = i
					if m != k:
						A[k][:], A[m][:] = A[m][:], A[k][:]
					break
		
		if A[k][:] != [0]*K:
			# Computing row operations
			for u in range(n):
				if A[k][u] != 0:
					break
			for x in range(q):
				if x*A[k][u] % q == 1:
					for l in range(u,n):
						A[k][l] = (A[k][l]*x) % q
					break
			for i in range(k+1,K):
				for j in range(u+1,n):
					A[i][j] = (A[i][j]-A[i][u]*A[k][j]) % q
				A[i][u] = 0

	# Eliminating duplicate rows
	for i in range(0,K):
		for j in range (0,K):
			if A[j][:]==A[i][:] and j!=i:
				A[i][:] = [0 for i in range(0,n)]
		
	# Determining basis
	BS = []
	for i in range(0,K):
		if A[i][:] != [0 for i in range(0,n)]:
			s = ''
			for k in range(0,n):
				s += str(A[i][k])
			BS.append(s)
	
	# Returning Output
	basis = "Basis obtained by Algorithm 4.1:\n{"+str(BS[0])
	for i in range(1,len(BS)):
		basis += ", "+str(BS[i])
	basis += "}"
	output1.config(text=basis)
  
#####################################################
# Algorithm 4.2

def algorithm_42():
	'''
	Performing Algorithm_4.2 to obtain a basis
	q: the number of alphabets in code
	S: an array of codewords
	'''
	q = int(q_input.get())
	S = list(map(str,S_input.get().split(',')))
	
	K = len(S)
	n = len(S[1])

	A = [[0 for k in range(0,K)] for i in range(0,n)]

	for i in range(0,n):
		for k in range (0,K):
			A[i][k] = int(S[k][i])

	## Finding REF
	p = min(n,K)
	for k in range(n):
		if k < p:
			for j in range(k,n):
				# Swapping two rows to obtain 1 as leading entry
				if A[j][k]!=1:
					m = k
					for i in range(k+1,n):
						if A[i][k] > A[m][k]:
							m = i
					if m != k:
						A[k][:], A[m][:] = A[m][:], A[k][:]
					break

		if A[k][:] != [0]*K:
			# Computing row operations
			for u in range(K):
				if A[k][u] != 0:
					break
			for x in range(q):
				if x*A[k][u] % q == 1:
					for l in range(u,K):
						A[k][l] = (A[k][l]*x) % q
					break
		for i in range(k+1,n):
			for j in range(u+1,K):
				A[i][j] = (A[i][j]-A[i][u]*A[k][j]) % q
			A[i][u] = 0

	# Eliminating duplicate rows
	for i in range(0,n):
		for j in range (0,n):
			if A[j][:]==A[i][:] and j!=i:
				A[i][:] = [0 for i in range(0,K)]

	# Determining basis
	BS = []

	u = 0
	for i in range(n):
		for v in range(u,K):
			if A[i][v]!=0:
				BS.append(S[v])
				u += 1
				break

	# Returning Output
	basis = "Basis obtained by Algorithm 4.2:\n{"+str(BS[0])
	for i in range(1,len(BS)):
		basis += ", "+str(BS[i])
	basis += "}"
	output1.config(text=basis)

#####################################################
# Algorithm 4.3

def algorithm_43():
	'''
	Performing Algorithm_4.3 to obtain a basis
	q: the number of alphabets in code
	S: an array of codewords
	'''
	q = int(q_input.get())
	S = list(map(str,S_input.get().split(',')))
	
	K = len(S)
	n = len(S[1])

	A = str_set_to_matrix(S)
	
	## Finding RREF
	p = min(n,K)
	for k in range(K):
		if k < p:
			for j in range(k,K):
				# Swapping two rows to obtain 1 as leading entry
				if A[j][k]!=1:
					m = k
					for i in range(k+1,K):
						if A[i][k] > A[m][k]:
							m = i
					if m != k:
						A[k][:], A[m][:] = A[m][:], A[k][:]
					break		

		
		if A[k][:] != [0]*K and A[k-1][:] != [0]*(n-1)+[1]:
			# Computing row operations
			for u in range(n):
				if A[k][u] != 0:
					break
			for x in range(q):
				if x*A[k][u] % q == 1:
					for l in range(u,n):
						A[k][l] = (A[k][l]*x) % q
					break
			for i in range(K):
				if i != k:
					for j in range(u+1,n):
						A[i][j] = (A[i][j]-A[i][u]*A[k][j]) % q
					A[i][u] = 0

	# Initializing a generator matrix
	i = 0
	G = []
	while i<p and A[i] != [0]*n:
		G.append(A[i])
		i += 1
	
	P = []
	for k in range(0,len(G)):
		if G[k][k] == 0:
			for u in range(k,n):
				# Swapping two columns to obtain 1 as leading entry
				if G[k][u]==1:
					for j in range(0,len(G)):
						G[j][k], G[j][u] = G[j][u], G[j][k]
					break
			P.append([k,u])

	# Creating a matrix - X transpose
	minXt = [[0 for i in range(len(G))] for j in range(n-len(G))]
	for i in range(len(G)):
		for j in range(n-len(G)):
			minXt[j][i] = - G[i][len(G)+j] % q

	# Creating a parity-check matrix H
	I = [[0]*i + [1] + [0]*(n-len(G) - i - 1) for i in range(n-len(G))]
	H = [[0 for j in range(n)] for i in range(n-len(G))]

	for i in range(n-len(G)):
		for j in range(n):
			if j<len(G):
				H[i][j] = minXt[i][j]
			elif n-len(G)!=0:
				H[i][j] = I[i][j-len(G)]
	
	# Permuting columns of matrix H
	if P != []:
		for k in range(1,len(P)):
			for j in range(len(H)):
				H[j][P[-k][0]], H[j][P[-k][1]] = H[j][P[-k][1]], H[j][P[-k][0]]
		for j in range(0,len(H)):
			H[j][P[0][0]], H[j][P[0][1]] = H[j][P[0][1]], H[j][P[0][0]]
	
	# Determining a dual basis
	DBS = []
	for i in range(n-len(G)):
		s = ''
		for k in range(n):
			s += str(H[i][k])
		DBS.append(s)
	
	# Returning Output
	dual_basis = "Dual basis obtained by Algorithm 4.3:\n{"+str(DBS[0])
	for i in range(1,len(DBS)):
		dual_basis += ", "+str(DBS[i])
	dual_basis += "}"
	output1.config(text=dual_basis)

#####################################################

def RREF(q,A):
	'''
	Computing the reduced row echelon form (RREF)
	q: the number of alphabets in code
	A: a matrix
	'''
	n = len(A[0])
	K = len(A)

	p = min(n,K)
	for k in range(K):
		if k < p:
			for j in range(k,K):
				# Swapping two rows to obtain 1 as leading entry
				if A[j][k]!=1:
					m = k
					for i in range(k+1,K):
						if A[i][k] > A[m][k]:
							m = i
					if m != k:
						A[k][:], A[m][:] = A[m][:], A[k][:]
					break		

		
		if A[k][:] != [0]*K and A[k-1][:] != [0]*(n-1)+[1]:
			# Computing row operations
			for u in range(n):
				if A[k][u] != 0:
					break
			for x in range(q):
				if x*A[k][u] % q == 1:
					for l in range(u,n):
						A[k][l] = (A[k][l]*x) % q
					break
			for i in range(K):
				if i != k:
					for j in range(u+1,n):
						A[i][j] = (A[i][j]-A[i][u]*A[k][j]) % q
					A[i][u] = 0
	return A

#####################################################

def generator_matrix(q,A):
	'''
	Computing the generator matrix of A
	q: the number of alphabets in code
	A: a matrix
	'''
	A = RREF(q,A)
	n = len(A[0])
	K = len(A)
	p = min(n,K)
	Z = [0 for k in range(0,n)]

	# Initializing matrix G
	i = 0
	G = []
	while i<p and A[i] != Z:
		G.append(A[i])
		i += 1
  
	return G

#####################################################

def parity_check_matrix(q,G):
	'''
	Computing the parity-check matrix of A
	q: the number of alphabets in code
	A: a matrix
	'''
	n= len(G[0])

	P = []
	for k in range(0,len(G)):
		if G[k][k] == 0:
			for u in range(k,n):
				# Swapping two columns to obtain 1 as leading entry
				if G[k][u]==1:
					for j in range(0,len(G)):
						G[j][k], G[j][u] = G[j][u], G[j][k]
					break
			P.append([k,u])

	# Creating matrix - X transpose
	minXt = [[0 for i in range(len(G))] for j in range(n-len(G))]
	for i in range(len(G)):
		for j in range(n-len(G)):
			minXt[j][i] = - G[i][len(G)+j] % q

	# Creating a parity-check matrix H
	I = [[0]*i + [1] + [0]*(n-len(G) - i - 1) for i in range(n-len(G))]
	H = [[0 for j in range(n)] for i in range(n-len(G))]

	for i in range(n-len(G)):
		for j in range(n):
			if j<len(G):
				H[i][j] = minXt[i][j]
			elif n-len(G)!=0:
				H[i][j] = I[i][j-len(G)]
	
	# Permuting columns of matrix H
	if P != []:
		for k in range(1,len(P)):
			for j in range(len(H)):
				H[j][P[-k][0]], H[j][P[-k][1]] = H[j][P[-k][1]], H[j][P[-k][0]]
		for j in range(0,len(H)):
			H[j][P[0][0]], H[j][P[0][1]] = H[j][P[0][1]], H[j][P[0][0]]

	return H
	
#####################################################

def encode():
	'''
	Encoding a codeword
	q: the number of alphabets in code
	U: a codeword string
	S: an array of codewords
	'''
	q = int(q2_input.get())
	U = str_set_to_matrix(u_input.get())
	S = str_set_to_matrix(list(map(str,S2_input.get().split(','))))
	
	A = RREF(q,S)
	if len(U) != len(A):
		output2.config(text="The length of 'Word' does not match.")
	else:
		G = generator_matrix(q,A)
		
		rows = len(G)
		colomns = len(G[0])
		
		P = []
		for k in range(0,min(rows,colomns)):
			if G[k][k] == 0:
				for u in range(k,colomns):
					# Swapping two columns to obtain 1 as leading entry
					if G[k][u]==1:
						for j in range(0,rows):
							G[j][k], G[j][u] = G[j][u], G[j][k]
						break
				P.append([k,u])
		
		encoded = [0 for j in range(colomns)]
		for j in range(colomns):
			for i in range(rows):
				encoded[j] += U[i][0]*G[i][j]
			encoded[j] = encoded[j] % q
		
		output2.config(text=encoded)

#####################################################

def Hamming_weight(u):
	'''
	Computing the Hamming weight of a codeword u
	u: a codeword string
	'''
	wt = 0
	for i in range(len(u)):
		if int(u[i]) != 0:
			wt += 1
	return wt

#####################################################

def syndrome_table(q,n,H):
	'''
	Computing the Hamming weight of a codeword u
	q: the number of alphabets
	n: the length of codewords in linear code C
	H: a parity-check matrix
	'''
	cosets = []
	for weight in range(len(H[0])):
		if len(cosets) == q*(len(H)):
			break
		codewords = generate_codewords(q,n,weight)
		for codeword in codewords:
			syndrome = [0]*len(H)
			for i in range(len(H)):
				for j in range(len(H[0])):
					syndrome[i] += codeword[j]*H[i][j]
				syndrome[i] = syndrome[i] % q
			if any(sub[1] == syndrome for sub in cosets) == False:
				cosets.append([codeword,syndrome])
	return cosets
  
#####################################################

def decode():
	'''
	Decoding an encoded codeword
	q: the number of alphabets in code
	v: an ecoded codeword string
	C: a linear code matrix
	'''
	q = int(q2_input.get())
	v = str_set_to_matrix(v_input.get())
	C = str_set_to_matrix(list(map(str,S2_input.get().split(','))))
	
	A = RREF(q,C)
	
	if len(v) != len(A[0]):
		output3.config(text="The length of 'Sent codeword' does not match.")
	else:
		n = len(C[0])
		K = len(C)
		G = generator_matrix(q,A)
		H = parity_check_matrix(q,G)

		syndrome_tab = syndrome_table(q,n,H)
		
		received_syndrome = [0]*len(H)
		for i in range(len(H)):
			for j in range(len(H[0])):
				received_syndrome[i] += v[j][0]*H[i][j]
			received_syndrome[i] = received_syndrome[i] % q

		for i in range(len(syndrome_tab)):
			if received_syndrome == syndrome_tab[i][1]:
				received = [0]*n
				for j in range(n):
					received[j] = (v[j][0] - syndrome_tab[i][0][j]) % q
				break

		decoded = received[0:len(C)]
		output3.config(text=decoded)

#####################################################

def generate_codewords(q,n,weight):
	'''
	Generating codewords of the least Hamming weight recursively
	q: the number of alphabets
	n: the length of codewords in linear code C
	weight: Hamming weight
	'''
	if weight > n:
		return []
	if n == 1:
		codewords = []
		for i in range(1,q):
			codewords.append([i])
		return codewords if weight == 1 else [[0]]
	if weight == 0:
		return [[0]*n]
	if weight == n:
		return [[1]*n]
  
	codewords = []
	for j in range(n):
		for codeword in generate_codewords(q,n-j-1,weight-1):
			for i in range(q):
				new_codeword1 = [0]*n
				new_codeword1[j] = i
				new_codeword1[j+1:j+1+len(codeword)] = codeword
				if Hamming_weight(new_codeword1) == weight and new_codeword1 not in codewords:
					codewords.append(new_codeword1)
				new_codeword2 = [0]*n
				new_codeword2[n-1-j] = i
				new_codeword2[n-1-j-len(codeword):n-1-j] = codeword
				if Hamming_weight(new_codeword2) == weight and new_codeword2 not in codewords:
					codewords.append(new_codeword2)
	return codewords
  
#####################################################

window.mainloop()

