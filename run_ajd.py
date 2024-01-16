from ajd import discrete1_bond
from ajd import discrete2_bond
from ajd import discrete3_bond
from ajd import exact_bond
from ajd import discrete1_cap
from ajd import discrete2_cap
from ajd import discrete3_cap
from ajd import exact_cap_final

n_sim = 500000
sim_frames = [10000, 20000, 40000, 100000, 500000]
all_results = []

print('----- AJD Model Tests -----')
print('--- Zero Bond Tests ---')
print('\nExact method: ')
data = exact_bond.monte_carlo(n_sim=n_sim, sim_frames=sim_frames)
all_results.append(data)
print(data)
with open('ajd_results.txt', 'a') as f:
    f.write('\n%s' % data)
print('\nDiscretization method I: ')
data = discrete1_bond.monte_carlo(sim_frames=sim_frames)
all_results.append(data)
print(data)
with open('ajd_results.txt', 'a') as f:
    f.write('\n%s' % data)
print('\nDiscretization method II: ')
data = discrete2_bond.monte_carlo(sim_frames=sim_frames)
all_results.append(data)
print(data)
with open('ajd_results.txt', 'a') as f:
    f.write('\n%s' % data)
print('\nDiscretization method III: ')
data = discrete3_bond.monte_carlo(sim_frames=sim_frames)
all_results.append(data)
print(data)
with open('ajd_results.txt', 'a') as f:
    f.write('\n%s' % data)

print('\n--- Cap Tests ---')
print('\nExact method: ')
data = exact_cap_final.monte_carlo(n_sim=1000000, sim_frames=[10000, 20000, 40000, 100000, 500000, 1000000])
all_results.append(data)
print(data)
with open('ajd_results.txt', 'a') as f:
    f.write('\n%s' % data)
print('\nDiscretization method I: ')
data = discrete1_cap.monte_carlo(sim_frames=sim_frames)
all_results.append(data)
print(data)
with open('ajd_results.txt', 'a') as f:
    f.write('\n%s' % data)
print('\nDiscretization method II: ')
data = discrete2_cap.monte_carlo(sim_frames=sim_frames)
all_results.append(data)
print(data)
with open('ajd_results.txt', 'a') as f:
    f.write('\n%s' % data)
print('\nDiscretization method III: ')
data = discrete3_cap.monte_carlo(sim_frames=sim_frames)
all_results.append(data)
print(data)
with open('ajd_results.txt', 'a') as f:
    f.write('\n%s' % data)

