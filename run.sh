mode=$1
mode=`echo $mode | tr '[:upper:]' '[:lower:]'`

if [[ $mode == 'pgd_both_sample' ]] || [[ $mode == 'pgd_both_robustness_sample' ]]
then
    python main.py --method $mode --epochs 300 --pgd_epsilon 0.005 --pgd_limit 0.075
elif [[ $mode == 'pgd_both_metric' ]] || [[ $mode == 'pgd_both_robustness_metric' ]]
then
    python main.py --method $mode --epochs 300 --pgd_epsilon 0.005 --pgd_limit 0.075 --batch_size 12
elif [[ $mode == 'gan_both_train' ]]
then
    python main.py --method $mode --batch_size 7 --log_interval 250 --epochs 500000
elif [[ $mode == 'gan_both_sample' ]] || [[ $mode == 'gan_both_robustness_sample' ]]
then
    python main.py --method $mode --gan_test_models "gan_both.pth"
elif [[ $mode == 'gan_both_metric' ]] || [[ $mode == 'gan_both_robustness_metric' ]]
then
    python main.py --method $mode --gan_test_models "gan_both.pth" --batch_size 14
elif [[ $mode == 'worker' ]]
then
    python main.py --method $mode --batch_size 64
elif [[ $mode == 'robustness' ]]
then
    python main.py --method $mode --batch_size 14
else
    echo "Unrecognized mode!"
fi