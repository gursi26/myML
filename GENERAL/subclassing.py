class Student :
	def __init__(self, name, age, grade, family_size, sibilings = False) :
		self.name = name
		self.age = age
		self.grade = grade
		self.family_size = family_size
		self.sibilings = sibilings

class IB_student(Student, object):
	def __init__(self, name, family_size, sibilings):
		super(IB_student, self).__init__(name, 27, 'IB year 1', family_size, sibilings)

myself = IB_student(name = 'Some name', family_size = 35, sibilings = True)
print(myself.age)