using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class OutOfTrack1 : MonoBehaviour
{
    // Start is called before the first frame update
    void OnTriggerEnter()
    {
        Debug.Log("Entered the Trigger");
        Global.StartTimer = false;
        SceneManager.LoadScene("test_sence");
    }


}
