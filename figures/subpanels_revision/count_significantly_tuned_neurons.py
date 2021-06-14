from figures.subpanels_revision import tuning_of_highly_weighted_neurons


def find_high_weighted_neurons(strain, recording):
    print('Running tuning of highly weighted neurons for ', strain, recording)
    tuning_of_highly_weighted_neurons.main(strain=strain, recording='BrainScanner'+recording, behavior='velocity')
    tuning_of_highly_weighted_neurons.main(strain=strain, recording='BrainScanner'+recording, behavior='curvature')


if True:
    strain = 'AKS297.51_moving'
    find_high_weighted_neurons(strain, '20200130_110803') #AML_310_A
    find_high_weighted_neurons(strain, '20200130_105254') #AML_310_B
    find_high_weighted_neurons(strain, '20200310_142022') #AML_310_C
    find_high_weighted_neurons(strain, '20200310_141211') #AML_310_D

    strain = 'AML32_moving'
    find_high_weighted_neurons(strain, '20170424_105620')  # AML32_AK
    find_high_weighted_neurons(strain, '20170610_105634')  # AML32_B
    find_high_weighted_neurons(strain, '20170613_134800')  # AML32_C
    find_high_weighted_neurons(strain, '20180709_100433')  # AML32_D
    find_high_weighted_neurons(strain, '20200309_151024')  # AML32_E
    find_high_weighted_neurons(strain, '20200309_153839')  # AML32_F
    find_high_weighted_neurons(strain, '20200309_162140')  # AML32_G

strain = 'AML18_moving'
find_high_weighted_neurons(strain,'20200116_145254') #AML18_A
find_high_weighted_neurons(strain,'20200116_152636') #AML18_B
find_high_weighted_neurons(strain,'20200204_102136') #AML18_C
find_high_weighted_neurons(strain,'20200310_153952') #AML18_D
find_high_weighted_neurons(strain,'20200311_100140') #AML18_E
find_high_weighted_neurons(strain,'20200929_140030') #AML18_F
find_high_weighted_neurons(strain,'20200929_143439') #AML18_G
find_high_weighted_neurons(strain,'20210503_122703') #AML18_H
find_high_weighted_neurons(strain,'20210503_135244') #AML18_I
find_high_weighted_neurons(strain,'20210503_151831') #AML18_J
find_high_weighted_neurons(strain,'20210503_154404') #AML18_K
