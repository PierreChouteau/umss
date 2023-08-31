'''run evaluations on multiple models'''

import os

models_to_evaluate = [
        #--------- Test des modèles 'originaux'
        # 'unsupervised_2s_bc1song_mf0_1',
        # 'unsupervised_4s_satb_bcbq_mf0_1_again',
        # '4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_bis',
        # '4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_bis',
        # '4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_rangeFreq',
        # '4s_cuestaTrainedNotFix_assignTrainedNotFix_BC1song_bis_bis',
        # 'unsupervised_4s_satb_bcbq_mf0_1_again',
        
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_bis_10epo",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_bis_10epo_bis",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_bis_15epo",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testsigmoid1",
        
        # '4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testreconstruction',
        # '4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testreconstruction1',
        # '4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testreconstruction2',
        # '4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testreconstruction3',
        
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testsigmoid_avecloss",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testsigmoid_avecloss1",

        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testBatchNorm",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testBatchNorm1",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testBatchNorm2",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_testBatchNorm3",
        
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeEval_DoubleLoss_lr",
        
        #---------  Test pour savoir s'il faut utilisé le mode train ou non
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeTrain",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeTrain_DoubleLoss",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeTrain_LossSalience",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeTrain_LossVoices",
        
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeEval_DoubleLoss_bis",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeEval_bis"
        
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeEval_rec",
        
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeEval_DoubleLoss",
        
        
        #--------- Test des moèles avec toute partie extraction des f0s fixe
        # "4s_cuestaTrainedFix_assignTrainedFix_BCBQ_ModeEval",
        # "4s_cuestaTrainedFix_assignTrainedFix_BCBQ_ModeEval_rec",
        # "4s_cuestaTrainedFix_assignTrainedFix_BC1song",
        # "4s_cuestaTrainedFix_assignTrainedFix_BC1song_ModeEval_DoubleLoss",
        
                
        #---------  Test des modèles sur BCBQ en mode Eval et avec différentes loss ajoutées
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_ModeEval_DoubleLoss_bis",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_ModeEval_NoLoss",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_ModeEval_LossSalience",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_ModeEval_LossVoices",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_ModeEval",
        
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_ModeEval_rec_DoubleLoss",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_ModeEval_rec_DoubleLoss_bis",
        
        
        #---------  Evaluation de modèle en mode Eval, principe de reconstruction avec ajout de la commitment loss - learning rate et valeur des loss changeantes
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeEval_rec_commitment_loss",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeEval_rec_commitment_loss_bis",
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BC1song_ModeEval_rec_commitment_loss_bis_bis",
        
        # "4s_cuestaTrainedFix_assignTrainedNotFix_BCBQ_ModeEval_rec_commitment_loss",
        
        #--------- Evaluation pour le full model entrainable sur BC1song
        # "4s_cuestaTrainedNotFix_assignTrainedNotFix_BC1song_rec_ModeEval_DoubleLoss",
        # "4s_cuestaTrainedNotFix_assignTrainedNotFix_BCBQ_rec_ModeEval_DoubleLoss_lr",
        
        #-------- Evaluation pour le full model entrainable sur BC1song mais en prenant les f0s de la mixture
        # "test",
        
        
        #--------- EVALUATION IMPORTANTE DU RAPPORT
        # BC1song
        # "4s_cuestaTrainedFix_assignTrainedFix_BCBQ_ModeEval_rec",
        # "CuestaNotFix_VANotFix_rec_modifie_double_loss_high_weight_BC1song",
        # "CuestaFix_VANotFix_rec_modifie_double_loss_high_weight_BC1song",
        # "10_epo_real_training_Full_NotFix_rec_modif_double_loss_BC1song_warmup_synth_BCBQ_after_warmup_F0_BC1song",
        
        
        # BCBQ
        "4s_cuestaTrainedFix_assignTrainedFix_BCBQ_ModeEval_rec",
        # "CuestaNotFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ",
        # "CuestaFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ_bis",
        # "1_epo_real_training_Full_NotFix_rec_modif_double_loss_BCBQ_warmup_synth_BCBQ_after_warmup_F0_BCBQ_avec_committment_loss_prolonge_bis",
        
        
        
        #-------- Evaluation pas classée.... désolé
        # "rec_unsupervised",
        # "rec_unsupervised_bis",
        # 'ste_unsupervised_bis',
        # "ste_unsupervised_lr0.00005",
        # "ste_supervised",
        # "ste_unsupervised_BCBQ",
        # "ste_unsupervised_bis_voice_unvoice",
        
        # "test_after_warm_up",
        # "warmup_F0",
        # "warmup_F0_BCBQ",
        # "warmup_F0_avec_committment_loss",
        # "warmup_F0_avec_committment_loss_prolonge",
        # "warmup_F0_BCBQ_avec_committment_loss_prolonge",
        # "warmup_synth_after_warmup_F0",
        # "warmup_synth_after_warmup_F0_avec_voice_unvoice",
        # "warmup_F0_after_warmup_synth_BCBQ",
        # "warmup_synth_BCBQ_after_warmup_F0_BC1song",
        # "CuestaFix_VANotFix_ste_double_loss_high_weight_BC1song",
        # "warmup_synth_BCBQ_after_warmup_F0_BC1song_prolonge",
        
        # "CuestaFix_VANotFix_ste_double_loss_high_weight__hardtanh_1000_BC1song",
        # "CuestaFix_VANotFix_ste_double_loss_normal_weight__hardtanh_1000_BC1song",
        
        # "real_training_after_warmup_synth_BCBQ_after_warmup_F0_BC1song",
        # "warmup_synth_BCBQ_after_warmup_F0_BC1song",
        
        # Evaluation des modèles avec reconstruction qui englobe le masque fréquentiel
        # "CuestaFix_VANotFix_rec_modifie_double_loss_high_weight_BC1song",
        # "CuestaFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BC1song",
        # "CuestaNotFix_VANotFix_rec_modifie_double_loss_high_weight_BC1song",
        # "CuestaNotFix_VANotFix_rec_modifie_double_loss_high_weight_BCBQ",
        # "CuestaNotFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ",
        # "CuestaNotFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BC1song",
        
        # "CuestaFix_VANotFix_rec_modifie_double_loss_high_weight_BCBQ_bis",
        # "CuestaFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ_bis",
        # "Full_not_Fix_after_CuestaFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ_bis",
        # "Full_not_Fix_bis_after_CuestaFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ_bis",
        
        # "real_training_BCBQ_after_warmup_synth_BCBQ_after_warmup_F0_BC1song_rec_modif_double_loss_sur_f0",
        # "real_training_BC1song_after_warmup_F0_BCBQ_after_warmup_synth_BCBQ_ste_pas_de_loss_sur_f0",
        # "real_training_BC1song_after_warmup_F0_BCBQ_after_warmup_synth_BCBQ_rec_modif_pas_de_loss_sur_f0",
        
        # "warmup_F0_avec_committment_loss_prolonge_bis",
        # "warmup_F0_BCBQ_avec_committment_loss_prolonge",
        # "warmup_synth_1s_bc1song",
        # "warmup_synth_BCBQ_after_warmup_F0_BCBQ_avec_committment_loss_prolonge_bis",
        # "real_training_rec_modif_double_loss_BCBQ_warmup_synth_BCBQ_after_warmup_F0_BCBQ_avec_committment_loss_prolonge_bis",
        # "real_training_Full_NotFix_rec_modif_double_loss_BCBQ_warmup_synth_BCBQ_after_warmup_F0_BCBQ_avec_committment_loss_prolonge_bis",
        # "real_training_Full_NotFix_rec_modif_double_loss_BC1song_warmup_synth_BCBQ_after_warmup_F0_BC1song",
        # "real_training_rec_modif_BCBQ_warmup_synth_BCBQ_after_warmup_F0_BCBQ_avec_committment_loss_prolonge_bis",
        # "1_epo_real_training_Full_NotFix_rec_modif_double_loss_BCBQ_warmup_synth_BCBQ_after_warmup_F0_BCBQ_avec_committment_loss_prolonge_bis",
        # "10_epo_real_training_Full_NotFix_rec_modif_double_loss_BC1song_warmup_synth_BCBQ_after_warmup_F0_BC1song",
        # 'CuestaFix_VANotFix_rec_double_loss_and_committment_high_weight_cantoria1song',
        
        # "test_suite_bis_1_epo_real_training_Full_NotFix_rec_modif_double_loss_BCBQ_warmup_synth_BCBQ_after_warmup_F0_BCBQ_avec_committment_loss_prolonge_bis",
        # "2Sources_warmup_synth_BCBQ_after_warmup_F0_BCBQ_avec_committment_loss_bis",
        
        # "CuestaNotFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ",
        # "CuestaFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ_bis"
        
        # "4s_Cuesta_NF_VA_NF_after_synth_mono_BCBQ_warmup_F0_BCBQ_avec_committment_loss_prolonge_1epo",
        # "4s_CuestaFix_VA_NF_after_synth_mono_BCBQ_warmup_F0_BCBQ_avec_committment_loss_prolonge",
        # "4s_Cuesta_NF_VA_NF_after_synth_mono_BCBQ_warmup_F0_BCBQ_avec_committment_loss_prolonge",
        # "Full_not_Fix_after_CuestaFix_VANotFix_rec_modifie_double_loss_and_committment_high_weight_BCBQ_bis",
        ]

eval_mode='default' # default evaluation
# eval_mode='fast' # fast evaluation
# eval_mode='robustness' # run many unique evaluations for each model, following different types of robustness tests
# eval_mode='robustness_vad' # run many unique evaluations for each model, with vad error tests only



for tag in models_to_evaluate:
    
    if eval_mode=='robustness_vad' :
        for i in range(5):
            command="python eval_robustness_tests.py --tag '{}' --f0-from-mix --test-set 'CSD' --teststocompute baseline_gtf0 gtf0_strict_error_percent --vadseed {}".format(tag,i)
            print(command)
            os.system(command)
    else:
        if eval_mode=='original_paper':
            command="python eval.py --tag '{}' --f0-from-mix --test-set 'CSD'".format(tag)
    
        elif eval_mode=='default':
            # Avec Crepe
            command="python eval.py --tag '{}' --test-set 'CSD' --show-progress --compute all".format(tag)
            # command="python eval.py --tag '{}' --test-set 'cantoria' --show-progress --compute all".format(tag)
            # Avec Cuesta
            # command="python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute all".format(tag)
        
        elif eval_mode=='fast':
            command="python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute SI-SDR_mask".format(tag)

        elif eval_mode=='robustness':
            command="python eval_robustness_tests.py --tag '{}' --f0-from-mix --test-set 'CSD' --teststocompute all".format(tag)

        elif eval_mode=='robustness_vad':
            command="python eval_robustness_tests.py --tag '{}' --f0-from-mix --test-set 'CSD' --teststocompute baseline_gtf0 gtf0_strict_error_percent --vadseed 0".format(tag)


        print(command)
        os.system(command)
