% %script to plot the different AVA correlation plots
% 
% % cd '/projects/LEIFER/PanNeuronal/20200130/BrainScanner20200130_110803_test_new'
% % figure
% % [corList_110803, RS_110803] = runCorrelation_AVANeuron(1, 1465, 33);
% % title('110803- Correaltion')
% % 
% % figure
% % bar(RS_110803)
% % title('110803- R^2')
% % xlabel('Neuron Number')
% % labelName = ['R^2 with AVA33'];
% % ylabel(labelName)
% % grid on 
% % xticks(0:5:length(RS_110803))
% % ylim([0 1])
% % 
% cd '/projects/LEIFER/PanNeuronal/20200313/BrainScanner20200313_102246_new_test'
% figure
% subplot(3, 1, 1)
% [corList_102246_move, RS_102246_move] = runCorrelation_AVANeuron(1, 1590, 30);
% title('102246-Correlation Moving')
% ylim([-1 1])
% xlim([0 125])
% 
% subplot(3,1,2)
% [corList_102246_immobile,RS_102246_immobile] = runCorrelation_AVANeuron(1700, 4016, 30);
% title('102246- Correlation Immobile')
% ylim([-1 1])
% xlim([0 125])
% 
% subplot(3,1,3)
% bar(corList_102246_move-corList_102246_immobile)
% title('Moving-Immobile')
% xlabel('Neuron Number')
% ylabel('Residual')
% xticks(0:5:length(corList_102246_move))
% grid on
% xlim([0 125])
% 
% figure
% subplot(3,1,1)
% bar(RS_102246_move)
% title('102246- R^2 Moving')
% xlabel('Neuron Number')
% labelName = ['R^2 with AVA30'];
% ylabel(labelName)
% grid on 
% xticks(0:5:length(RS_102246_move))
% ylim([0 1])
% xlim([0 125])
% 
% subplot(3,1,2)
% bar(RS_102246_immobile)
% title('102246- R^2 Moving')
% xlabel('Neuron Number')
% labelName = ['R^2 with AVA30'];
% ylabel(labelName)
% grid on 
% xticks(0:5:length(RS_102246_immobile))
% ylim([0 1])
% xlim([0 125])
% 
% subplot(3,1,3)
% bar(RS_102246_move-RS_102246_immobile)
% title('102246- R^2 Residuals')
% xlabel('Neuron Number')
% labelName = ['Residuals with AVA30'];
% ylabel(labelName)
% grid on 
% xticks(0:5:length(RS_102246_move))
% xlim([0 125])
% 
% 
% figure
% subplot(3, 1, 1)
% [corList_102246_move, RS_102246_move] = runCorrelation_AVANeuron(1, 1590, 18);
% title('102246-Correlation Moving')
% ylim([-1 1])
% xlim([0 125])
% xlim([0 125])
% 
% subplot(3,1,2)
% [corList_102246_immobile, RS_102246_immobile] = runCorrelation_AVANeuron(1700, 4016, 18);
% title('102246- Correlation Immobile')
% ylim([-1 1])
% xlim([0 125])
% 
% subplot(3,1,3)
% bar(corList_102246_move-corList_102246_immobile)
% title('Moving-Immobile')
% xlabel('Neuron Number')
% ylabel('Residual')
% xticks(0:5:length(corList_102246_move))
% grid on
% xlim([0 125])
% 
% figure
% subplot(3,1,1)
% bar(RS_102246_move)
% title('102246- R^2 Moving')
% xlabel('Neuron Number')
% labelName = ['R^2 with AVA18'];
% ylabel(labelName)
% grid on 
% xticks(0:5:length(RS_102246_move))
% ylim([0 1])
% xlim([0 125])
% 
% subplot(3,1,2)
% bar(RS_102246_immobile)
% title('102246- R^2 Moving')
% xlabel('Neuron Number')
% labelName = ['R^2 with AVA18'];
% ylabel(labelName)
% grid on 
% xticks(0:5:length(RS_102246_immobile))
% ylim([0 1])
% xlim([0 125])
% 
% subplot(3,1,3)
% bar(RS_102246_move-RS_102246_immobile)
% title('102246- R^2 Residuals')
% xlabel('Neuron Number')
% labelName = ['Residuals with AVA18'];
% ylabel(labelName)
% grid on 
% xticks(0:5:length(RS_102246_move))
% xlim([0 125])

cd '/projects/LEIFER/PanNeuronal/20200317/test_BrainScanner20200317_111126'
figure
subplot(3, 1, 1)
[corList_111126_move, RS_111126_move] = runCorrelation_AVANeuron(1, 1785, 28);
title('111126-Correlation Moving')
ylim([-1 1])
xlim([0 115])

subplot(3,1,2)
[corList_111126_immobile, RS_111126_immobile] = runCorrelation_AVANeuron(1900, 3360, 28);
title('111126- Correlation Immobile')
ylim([-1 1])
xlim([0 115])

subplot(3,1,3)
bar(corList_111126_move-corList_111126_immobile)
title('Moving-Immobile')
xlabel('Neuron Number')
ylabel('Residual')
xticks(0:5:length(corList_111126_move))
grid on 
xlim([0 115])

figure
subplot(3,1,1)
bar(RS_111126_move)
title('111126- R^2 Moving')
xlabel('Neuron Number')
labelName = ['R^2 with AVA28'];
ylabel(labelName)
grid on 
xticks(0:5:length(RS_111126_move))
ylim([0 1])
xlim([0 115])

subplot(3,1,2)
bar(RS_111126_immobile)
title('111126- R^2 Moving')
xlabel('Neuron Number')
labelName = ['R^2 with AVA28'];
ylabel(labelName)
grid on 
xticks(0:5:length(RS_111126_immobile))
ylim([0 1])
xlim([0 115])

subplot(3,1,3)
bar(RS_111126_move-RS_111126_immobile)
title('111126- R^2 Residuals')
xlabel('Neuron Number')
labelName = ['Residuals with AVA28'];
ylabel(labelName)
grid on 
xticks(0:5:length(RS_111126_move))
xlim([0 115])



