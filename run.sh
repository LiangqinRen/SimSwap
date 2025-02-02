mode=$1
mode=`echo $mode | tr '[:upper:]' '[:lower:]'`

if [ $# == 2 ] && [ "$2" = "console" ]; then
    console_only="--console_only"
else
    console_only=""
fi

if [[ $mode == 'pgd_both_sample' ]]
then
    python main.py $console_only --method $mode --epochs 1000 --pgd_epsilon 0.005
elif [[ $mode == 'pgd_both_robustness_sample' ]]
then
    python main.py $console_only --method $mode --epochs 335 --pgd_epsilon 0.005
elif [[ $mode == 'pgd_both_metric' ]]
then
    python main.py $console_only --method $mode --epochs 1000 --pgd_epsilon 0.005 --batch_size 12
elif [[ $mode == 'pgd_both_robustness_metric' ]]
then
    python main.py $console_only --method $mode --epochs 335 --pgd_epsilon 0.005 --batch_size 12
elif [[ $mode == 'pgd_robustness_forensics_sample' ]]
then
    python main.py $console_only --method $mode --epochs 335 --pgd_epsilon 0.005
elif [[ $mode == 'pgd_robustness_forensics_metric' ]]
then
    python main.py $console_only --method $mode --epochs 335 --pgd_epsilon 0.005 --batch_size 30
elif [[ $mode == 'gan_both_train' ]]
then
    python main.py $console_only --method $mode --batch_size 7 --log_interval 250 --epochs 500000
elif [[ $mode == 'gan_both_train_robust' ]]
then
    python main.py $console_only --method 'gan_both_train' --batch_size 7 --log_interval 250 --epochs 250000 --gan_train_robust --gan_test_models "gan_both.pth"
elif [[ $mode == 'gan_both_sample' ]] || [[ $mode == 'gan_both_robustness_sample' ]]
then
    python main.py $console_only --method $mode --gan_test_models "gan_both.pth"
elif [[ $mode == 'gan_both_metric' ]] || [[ $mode == 'gan_both_robustness_metric' ]]
then
    python main.py $console_only --method $mode --gan_test_models "gan_both.pth" --batch_size 12
elif [[ $mode == 'anchor_difference' ]]
then
    python main.py $console_only --method $mode --batch_size 64
elif [[ $mode == 'robustness' ]]
then
    python main.py $console_only --method $mode --batch_size 128
else
    echo "Unrecognized mode!"
fi