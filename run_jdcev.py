from jdcev import exact_put
from jdcev import exact_asian
from jdcev import exact_docall
from jdcev import discrete_put
from jdcev import discrete_asian
from jdcev import discrete_docall

n_sim = 500000
sim_frames = [10000, 20000, 40000, 100000, 500000]
all_results = []

print('----- JDCEV Model Tests -----')
# print('--- European Put Tests ---')
# print('\nExact method: ')
# data = exact_put.monte_carlo(n_sim=n_sim, sim_frames=sim_frames)
# all_results.append(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
# print(data)
# print('\nDiscretization method: ')
# data = discrete_put.monte_carlo(sim_frames=sim_frames)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
#
# print('\n--------------------------------------------\n')
# print('--- Asian Put Tests ---')
# print('\nExact method: ')
# data = exact_asian.monte_carlo(n_sim=n_sim, sim_frames=sim_frames)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
# print('\nDiscretization method: ')
# data = discrete_asian.monte_carlo(sim_frames=sim_frames)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)

print('\n--------------------------------------------\n')
print('--- Down-and-Out Call Tests ---')
# print('\nExact method: ')
# data = exact_docall.monte_carlo(n_sim=n_sim, sim_frames=sim_frames)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)

print('\nDiscretization method: ')
data = discrete_docall.monte_carlo(sim_frames=sim_frames)
all_results.append(data)
print(data)
with open('jdcev_results.txt', 'a') as f:
    f.write('\n%s' % data)

#
# print('\n- Alternate parameters -')
# print(' b = 0.2')
# print('Exact method: ')
# data = exact_put.monte_carlo(n_sim=n_sim, sim_frames=sim_frames, b=0.2)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
# print('\nDiscretization method: ')
# data = discrete_put.monte_carlo(sim_frames=sim_frames, b=0.2)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
#
# print('\n c = 1')
# print('Exact method: ')
# data = exact_put.monte_carlo(n_sim=n_sim, sim_frames=sim_frames, c=1)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
# print('\nDiscretization method: ')
# data = discrete_put.monte_carlo(sim_frames=sim_frames, c=1)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
#
# print('\n X_0 = 25')
# print('Exact method: ')
# data = exact_put.monte_carlo(n_sim=n_sim, sim_frames=sim_frames, X_0=25)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
# print('\nDiscretization method: ')
# data = discrete_put.monte_carlo(sim_frames=sim_frames, X_0=25)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
#
# print('\n beta = -0.5')
# print('Exact method: ')
# data = exact_put.monte_carlo(n_sim=n_sim, sim_frames=sim_frames, beta=-0.5)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)
# print('\nDiscretization method: ')
# data = discrete_put.monte_carlo(sim_frames=sim_frames, beta=-0.5)
# all_results.append(data)
# print(data)
# with open('jdcev_results.txt', 'a') as f:
#     f.write('\n%s' % data)

