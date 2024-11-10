mode=$1
mode=`echo $mode | tr '[:upper:]' '[:lower:]'`

if [[ $mode == 'pgd_both_sample' ]] || [[ $mode == 'pgd_both_robustness_sample' ]]
then
    python main.py --method $mode --epochs 300 --pgd_epsilon 0.005 --pgd_limit 0.075
elif [[ $mode == 'pgd_both_metric' ]] || [[ $mode == 'pgd_both_robustness_metric' ]]
then
    python main.py --method $mode --epochs 300 --pgd_epsilon 0.005 --pgd_limit 0.075 --batch_size 12
# elif [[ $mode == 'pgd_source_distance' ]]
# then
#     python main.py --method $mode --pgd_mimic "samples/zjl.jpg" --batch_size 50 --epochs 75 --pgd_epsilon 0.05 --pgd_limit 0.25 --log_interval 10
elif [[ $mode == 'gan_source_sample' ]] || [[ $mode == 'gan_target_sample' ]]  || [[ $mode == 'gan_source_robustness_sample' ]] || [[ $mode == 'gan_target_robustness_sample' ]]
then
    # python main.py --method $mode --gan_test_models "gan_source.pth"
    # python main.py --method $mode --gan_test_models "gan_source_no_RGB.pth"
    # python main.py --method $mode --gan_test_models "gan_source_no_middle.pth"
    # python main.py --method $mode --gan_test_models "gan_source_no_result.pth"
    # python main.py --method $mode --gan_test_models "gan_source_no_all.pth"
    python main.py --method $mode --gan_test_models "gan_target.pth"
    python main.py --method $mode --gan_test_models "gan_target_no_RGB.pth"
    python main.py --method $mode --gan_test_models "gan_target_no_middle.pth"
    python main.py --method $mode --gan_test_models "gan_target_no_result.pth"
    python main.py --method $mode --gan_test_models "gan_target_no_all.pth"
elif [[ $mode == 'gan_source_metric' ]] || [[ $mode == 'gan_target_metric' ]] || [[ $mode == 'gan_source_robustness_metric' ]] || [[ $mode == 'gan_target_robustness_metric' ]]
then
    python main.py --method $mode --gan_test_models "gan_target_no_all.pth" --batch_size 25
elif [[ $mode == 'worker' ]]
then
    python main.py --method $mode --batch_size 30
else
    echo "Unrecognized mode!"
fi