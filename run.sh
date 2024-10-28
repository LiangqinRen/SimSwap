mode=$1
mode=`echo $mode | tr '[:upper:]' '[:lower:]'`

if [[ $mode == 'swap' ]]
then
    python main.py --method $mode --swap_source "samples/dog.jpg" --swap_target "samples/zjl.jpg"
# elif [[ $mode == 'metric' ]] || [[ $mode == 'split' ]]
# then
#     python main.py --method $mode
# elif [[ $mode == 'pgd_source_single' ]]
# then
#     python main.py --method $mode --pgd_source "samples/zjl.jpg" --pgd_target "samples/zrf.jpg" --pgd_mimic "samples/james.jpg" --epochs 100 --pgd_epsilon 0.01 --pgd_limit 0.10
# elif [[ $mode == 'pgd_source_multi' ]]
# then
#     python main.py --method $mode --pgd_mimic "samples/james.jpg" 
# elif [[ $mode == 'pgd_source_metric' ]]
# then
#     python main.py --method $mode --pgd_mimic "samples/zjl.jpg" --batch_size 27 --epochs 75 --pgd_epsilon 0.01 --pgd_limit 0.125
# elif [[ $mode == 'pgd_source_distance' ]]
# then
#     python main.py --method $mode --pgd_mimic "samples/zjl.jpg" --batch_size 27 --epochs 50 --pgd_metric_people 100 --log_interval 10
# elif [[ $mode == 'pgd_target_single' ]]
# then
#     python main.py --method $mode --pgd_source "samples/zrf.jpg" --pgd_target "samples/hzxc.jpg" --pgd_mimic "samples/james.jpg" --pgd_limit 0.035
# elif [[ $mode == 'pgd_target_multi' ]]
# then
#     python main.py --method $mode --pgd_mimic "samples/james.jpg" --pgd_limit 0.035
# elif [[ $mode == 'pgd_target_metric' ]]
# then
#     python main.py --method $mode --pgd_mimic "samples/zjl.jpg" --batch_size 16 --epochs 50 
# elif [[ $mode == 'gan_source' ]] || [[ $mode == 'gan_target' ]]
# then
#     python main.py --method $mode --epochs 100000
# elif [[ $mode == 'gan_source_metric' ]]
# then
#     python main.py --method $mode --gan_test_models "gan_src.pth" --batch_size 10
elif [[ $mode == 'pgd_both_sample' ]]
then
    python main.py --method $mode --epochs 75 --pgd_epsilon 0.01 --pgd_limit 0.1 --pgd_mimic "samples/james.jpg"
elif [[ $mode == 'pgd_source_distance' ]]
then
    python main.py --method $mode --pgd_mimic "samples/zjl.jpg" --batch_size 50 --epochs 75 --pgd_epsilon 0.05 --pgd_limit 0.25 --log_interval 10
elif [[ $mode == 'pgd_source_sample' ]] || [[ $mode == 'pgd_source_robustness_sample' ]]
then
    python main.py --method $mode --pgd_mimic "samples/james.jpg" --epochs 100 --pgd_epsilon 0.15 --pgd_limit 0.5
elif [[ $mode == 'pgd_target_sample' ]] || [[ $mode == 'pgd_target_robustness_sample' ]]
then
    python main.py --method $mode --pgd_mimic "samples/james.jpg" --epochs 30 --pgd_epsilon 0.05 --pgd_limit 0.05
elif [[ $mode == 'pgd_source_metric' ]] || [[ $mode == 'pgd_source_robustness_metric' ]]
then
    python main.py --method $mode --pgd_mimic "samples/zjl.jpg" --epochs 75 --pgd_epsilon 0.05 --pgd_limit 0.25 --batch_size 27
elif [[ $mode == 'pgd_target_metric' ]]
then
    python main.py --method $mode --pgd_mimic "samples/zjl.jpg" --epochs 30 --pgd_epsilon 0.05 --pgd_limit 0.05 --batch_size 27
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
elif [[ $mode == 'test' ]]
then
    python main.py --method $mode --epochs 75 --pgd_epsilon 0.01 --pgd_limit 0.1 --pgd_mimic "samples/james.jpg"
else
    echo "Unrecognized mode!"
fi