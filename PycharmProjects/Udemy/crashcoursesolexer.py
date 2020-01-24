def fun1(x) :
    return x**4
x = 7
print(fun1(x))

s = "Hi there Sam!"
ssplt = list(s.split())
print(ssplt)

name ="Earth"
diameter = 12742
print("The diameter of {} is {}".format(name, diameter))

d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
print(d['k1'][3]['tricky'][3]['target'][3])

mail = "srijan.das0310@gmail.com"
domain_name = list(mail.split(sep = "@"))
print(domain_name[1])

def finddog(line) :
    words = line.split()
    if 'dog' in words :
        print("yes")
    else :
        print("no")
finddog("there is no dog here")

def countdogs(line) :
    words = line.split()
    dogs = 0
    for i in range(len(words)) :
        if words[i] == 'dog' :
            dogs = dogs + 1
    print ("Number of dogs were ", dogs)
countdogs("dog gog log dog mog dog")

seq = ['soup','dog','salad','cat','great']
new_seq = list(filter(lambda word : word[0]=='s' , seq))
print(new_seq)

def caught_speeding(speed, isBirthday) :
    if speed <= 60 :
        return "no ticket"
    else :
        if isBirthday == True :
            if speed >= 66 and speed <=86 :
                return "small ticket"
            else :
                return "big ticket"
        else :
            if speed >= 61 and speed <=81 :
                return "small ticket"
            else :
                return "big ticket"
ticket = caught_speeding(64, False)
print(ticket)