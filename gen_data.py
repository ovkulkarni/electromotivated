from coolname import generate_slug
import time
import datetime
import random

cols = ['Username', 'Project Name', 'Timestamp', '# of Components', '# of Batteries', '# of Resistors', '# of Capacitors', '# of Inductors', '# of Voltmeters', '# of Switches', 'Avg Voltage', 'Avg Current', 'Avg Resistance', 'Avg Power', '# of Views', '# of Comments', 'Avg Comment Length']
events = []

u_list = open('usernames.txt').read().splitlines()
for i in range(30000):
	events.append([random.choice(u_list)])

for i in range(30000):
	events[i].append(generate_slug())
	events[i].append(str(datetime.datetime.fromtimestamp(time.time() + random.randint(-1000000, 1000000)).strftime('%Y-%m-%d %H:%M:%S')))
	comp = [random.randint(0, 100) for _ in range(4)]
	events[i].append(sum(comp))
	events[i] += comp
	avg_cur = random.uniform(0, 100)
	avg_res = random.uniform(0, 100)
	avg_vol = avg_cur * avg_res + random.uniform(-5, 5)
	events[i].append(avg_vol)
	events[i].append(avg_cur)
	events[i].append(avg_res)
	events[i].append(avg_vol * avg_cur + random.uniform(-5, 5))
	events[i].append(random.randint(0, 10))
	if random.randint(0, 100) >= 90:
		events[i][-1] += random.randint(0, 100)
	events[i].append(random.randint(0, 3))
	events[i].append(random.uniform(0, 100))

print(';'.join(cols))
for row in events:
	print(';'.join(str(e) for e in row))